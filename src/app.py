import os
import cv2
import uuid
import threading
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from flask import Flask, render_template, request, send_file, jsonify

app = Flask(__name__, 
    static_url_path='',  
    static_folder='static',  
    template_folder='templates'
)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

executor = ThreadPoolExecutor(max_workers=4)
task_status = {}
task_lock = threading.Lock()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def resize_if_large(image, max_size=1024):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image

def process_image_task(image, params, task_id, output_path):
    try:
        with task_lock:
            task_status[task_id] = {
                'status': 'processing',
                'progress': 0,
                'output_path': output_path  
            }

        image = resize_if_large(image)

        with task_lock:
            task_status[task_id]['progress'] = 30

        enhanced = enhance_image_exposure(
            image, 
            gamma=params['gamma'],
            lambda_=params['lambda_'],
            dual=not params['use_lime']
        )

        with task_lock:
            task_status[task_id]['progress'] = 80

        cv2.imwrite(output_path, enhanced)

        with task_lock:
            task_status[task_id].update({
                'status': 'completed',
                'progress': 100,
                'output_path': output_path
            })

        return True
    except Exception as e:
        with task_lock:
            task_status[task_id] = {
                'status': 'failed',
                'error': str(e),
                'output_path': output_path
            }
        return False

def get_sparse_neighbor(p: int, n: int, m: int):
    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d

def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15):
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))
    return kernel

def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)

def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
                               bc: float = 1, bs: float = 1, be: float = 1):
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    return fused_images

def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    L_refined = np.clip(L_refined, eps, 1) ** gamma

    return L_refined

def correct_underexposure(im: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    L = np.max(im, axis=-1)
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d
    return im_corrected

def enhance_image_exposure(im: np.ndarray, gamma: float, lambda_: float, dual: bool = True, sigma: int = 3,
                         bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3):
    kernel = create_spacial_affinity_kernel(sigma, size=7)

    im_normalized = im.astype(np.float32) / 255.
    
    height, width = im_normalized.shape[:2]
    scale = 1
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        small_im = cv2.resize(im_normalized, None, fx=scale, fy=scale)
    else:
        small_im = im_normalized

    under_corrected = correct_underexposure(small_im, gamma, lambda_, kernel, eps)

    if dual:
        inv_im_normalized = 1 - small_im
        over_corrected = 1 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
        im_corrected = fuse_multi_exposure_images(small_im, under_corrected, over_corrected, bc, bs, be)
    else:
        im_corrected = under_corrected

    if scale != 1:
        im_corrected = cv2.resize(im_corrected, (width, height))

    return np.clip(im_corrected * 255, 0, 255).astype("uint8")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        task_id = str(uuid.uuid4())

        params = {
            'gamma': float(request.form.get('gamma', 0.6)),
            'lambda_': float(request.form.get('lambda', 0.15)),
            'use_lime': request.form.get('method', 'DUAL') == 'LIME'
        }
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'enhanced_{task_id}_{filename}')

        executor.submit(process_image_task, image, params, task_id, output_path)
        
        return jsonify({'task_id': task_id})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def get_task_status(task_id):
    with task_lock:
        status = task_status.get(task_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/result/<task_id>')
def get_result(task_id):
    try:
        with task_lock:
            status = task_status.get(task_id, {})
            if status.get('status') != 'completed':
                return jsonify({'error': 'Result not ready'}), 404
            
            output_path = status.get('output_path')
            if not output_path or not os.path.exists(output_path):
                return jsonify({'error': 'Result file not found'}), 404
            
            return send_file(output_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5500) 