import cv2
import numpy as np
import random
import time
from tqdm import tqdm

# Попытка импортировать CuPy. Если не получится, будет использован NumPy.
try:
    import cupy as cp
    GPU_ENABLED = True
    print("CuPy найден. Расчеты будут выполняться на GPU.")
except ImportError:
    cp = np
    GPU_ENABLED = False
    print("CuPy не найден. Расчеты будут выполняться на CPU (медленнее).")

def calculate_ssd_batch(patches_batch, target_patch):
    """
    Вычисляет SSD для целого батча патчей на GPU (или CPU).
    patches_batch: массив формы (N, H, W, C)
    target_patch: массив формы (H, W, C)
    """
    # xp - это либо cupy, либо numpy в зависимости от доступности
    xp = cp.get_array_module(patches_batch)
    
    # (N, H, W, C) -> (N, H*W*C)
    patches_flat = patches_batch.reshape(patches_batch.shape[0], -1)
    # (H, W, C) -> (H*W*C,)
    target_flat = target_patch.flatten()
    
    # Используем векторизованные операции для скорости
    diff = patches_flat - target_flat
    ssd_errors = xp.sum(diff * diff, axis=1)
    return ssd_errors

def find_min_cost_path_vertical(overlap_error):
    # Эта функция остается на CPU, так как она последовательна
    cost_matrix = np.zeros_like(overlap_error, dtype=np.float64)
    path_matrix = np.zeros_like(overlap_error, dtype=int)
    cost_matrix[0, :] = overlap_error[0, :]
    for i in range(1, overlap_error.shape[0]):
        for j in range(overlap_error.shape[1]):
            left, right = max(0, j-1), min(overlap_error.shape[1], j+2)
            prev = cost_matrix[i-1, left:right]
            cost_matrix[i, j] = overlap_error[i, j] + np.min(prev)
            path_matrix[i, j] = left + np.argmin(prev)
    path = np.zeros(overlap_error.shape[0], dtype=int)
    path[-1] = np.argmin(cost_matrix[-1])
    for i in range(overlap_error.shape[0]-2, -1, -1):
        path[i] = path_matrix[i+1, path[i+1]]
    return path

def find_min_cost_path_horizontal(overlap_error):
    return find_min_cost_path_vertical(overlap_error.T)

def synthesize_texture_gpu(source_im, out_size, patch_size, overlap, num_candidates=80):
    xp = cp if GPU_ENABLED else np
    h_src, w_src, _ = source_im.shape
    h_out, w_out = out_size

    # 1. Отправляем данные на GPU
    source_gpu = xp.asarray(source_im)
    canvas_gpu = xp.zeros((h_out, w_out, 3), dtype=xp.uint8)
    step = patch_size - overlap
    
    # Формируем банк патчей сразу на GPU
    src_patches_gpu = xp.array([source_gpu[sy:sy+patch_size, sx:sx+patch_size]
                                for sy in range(h_src-patch_size+1)
                                for sx in range(w_src-patch_size+1)])

    y_coords = range(0, h_out - patch_size + 1, step)
    for y in tqdm(y_coords, desc="Синтез на GPU"):
        for x in range(0, w_out - patch_size + 1, step):
            if x == 0 and y == 0:
                sel_patch_gpu = random.choice(src_patches_gpu)
            else:
                indices = np.random.choice(len(src_patches_gpu), min(num_candidates, len(src_patches_gpu)), replace=False)
                candidates_gpu = src_patches_gpu[indices]
                
                errors = xp.zeros(len(candidates_gpu))
                # 2. Выполняем расчеты батчами на GPU
                if x > 0:
                    target = canvas_gpu[y:y+patch_size, x:x+overlap]
                    candidates_overlap = candidates_gpu[:, :, :overlap]
                    errors += calculate_ssd_batch(candidates_overlap, target)
                if y > 0:
                    target = canvas_gpu[y:y+overlap, x:x+patch_size]
                    candidates_overlap = candidates_gpu[:, :overlap, :]
                    errors += calculate_ssd_batch(candidates_overlap, target)
                
                min_err = xp.min(errors)
                best_indices = xp.where(errors <= min_err * 1.05)[0]
                chosen_idx = random.choice(best_indices)
                sel_patch_gpu = candidates_gpu[chosen_idx]
            
            # 3. Поиск шва - временно возвращаем данные на CPU
            # Это узкое место, но его сложно избежать без полной переделки на CUDA
            mask_np = np.ones((patch_size, patch_size), dtype=bool)
            if x > 0:
                overlap_zone_canvas = cp.asnumpy(canvas_gpu[y:y+patch_size, x:x+overlap])
                overlap_zone_patch = cp.asnumpy(sel_patch_gpu[:, :overlap])
                ovl_v_err_np = np.sum((overlap_zone_canvas.astype(np.float32) - overlap_zone_patch.astype(np.float32))**2, axis=2)
                path = find_min_cost_path_vertical(ovl_v_err_np)
                for i in range(patch_size): mask_np[i, :path[i]] = False
            if y > 0:
                overlap_zone_canvas = cp.asnumpy(canvas_gpu[y:y+overlap, x:x+patch_size])
                overlap_zone_patch = cp.asnumpy(sel_patch_gpu[:overlap, :])
                ovl_h_err_np = np.sum((overlap_zone_canvas.astype(np.float32) - overlap_zone_patch.astype(np.float32))**2, axis=2)
                path = find_min_cost_path_horizontal(ovl_h_err_np)
                for i in range(patch_size): mask_np[:path[i], i] = False

            # Применяем маску на GPU
            mask_gpu = xp.asarray(mask_np)
            canvas_gpu[y:y+patch_size, x:x+patch_size] = xp.where(xp.stack([mask_gpu]*3, axis=-1), sel_patch_gpu, canvas_gpu[y:y+patch_size, x:x+patch_size])

    # 4. Возвращаем финальный результат с GPU на CPU
    return cp.asnumpy(canvas_gpu)


if __name__ == '__main__':
    input_file = '/input/texture.png'
    output_file = '/out/texture_tiled_gpu.jpg'
    
    source = cv2.imread(input_file)
    if source is None:
        print('Ошибка: Не удалось загрузить изображение. Создаю тестовую текстуру 1100x1100...')
        source = np.random.randint(0, 255, (1100, 1100, 3), dtype=np.uint8)
        cv2.imwrite(input_file, source)

    # Для такого большого изображения, patch_size лучше сделать побольше
    patch_size = 64
    overlap = int(patch_size * 0.65)
    
    start_time = time.time()
    result = synthesize_texture_gpu(source, (3300, 3300), patch_size, overlap)
    duration = time.time() - start_time
    
    cv2.imwrite(output_file, result)
    print(f'Готово! Изображение сохранено в {output_file}')
    print(f'Время выполнения: {duration/60:.2f} минут')