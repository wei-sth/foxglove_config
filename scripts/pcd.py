#!/usr/bin/env python3
# open3d can read, transform, etc, quite slow
# import sensor_msgs_py.point_cloud2 as pc2   pc2.read_points() is slow, use numpy instead

import logging
from collections import namedtuple
import os
import sys
import zlib
import struct
from sensor_msgs.msg import PointField
import numpy as np
import pandas as pd
import open3d as o3d
import util_func as uf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def colorize_by_value(values: np.ndarray, colormap='terrain'):
    """map value to rgb"""
    from matplotlib import cm
    min_val = np.percentile(values, 5)
    max_val = np.percentile(values, 95)
    # any value greater than your new max_val is treated as max_val
    clipped_values = np.clip(values, min_val, max_val)
    norm_z = (clipped_values - min_val) / (max_val - min_val)
    # norm_z = (values - values.min()) / (values.max() - values.min())
    if colormap == 'jet':
        colors = cm.jet(norm_z)[:, :3] * 255
    elif colormap == 'viridis':
        colors = cm.viridis(norm_z)[:, :3] * 255
    elif colormap == 'terrain':
        colors = cm.terrain(norm_z)[:, :3] * 255
    elif colormap == 'rviz':
        from matplotlib.colors import LinearSegmentedColormap
        rviz2_colors = [
        (0.00, (0.0, 0.0, 1.0)),  # 深蓝
        (0.05, (0.0, 0.5, 1.0)),  # 亮蓝
        (0.1, (0.0, 1.0, 1.0)),  # 青色
        (0.15, (0.5, 1.0, 0.5)),  # 黄绿
        (0.2, (1.0, 1.0, 0.0)),  # 亮黄
        (0.25, (1.0, 0.6, 0.0)),  # 橙色
        (0.3, (1.0, 0.1, 0.0)),   # 亮红（避免深红）
        (0.35, (0.0, 0.0, 1.0)),  # 深蓝
        (0.4, (0.0, 0.5, 1.0)),  # 亮蓝
        (0.45, (0.0, 1.0, 1.0)),  # 青色
        (0.5, (0.5, 1.0, 0.5)),  # 黄绿
        (0.55, (1.0, 1.0, 0.0)),  # 亮黄
        (0.95, (1.0, 0.6, 0.0)),  # 橙色
        (1.00, (1.0, 0.1, 0.0))   # 亮红（避免深红）
        ]
        custom_cmap = LinearSegmentedColormap.from_list(
            'rviz2_style', 
            [color for _, color in rviz2_colors]
            )
        colors = custom_cmap(norm_z)[:, :3] * 255
    elif colormap == 'rainbow':
        colors = cm.rainbow(norm_z)[:, :3] * 255
    else:
        colors = cm.plasma(norm_z)[:, :3] * 255
    return colors.astype(np.uint8)

def colorize_by_z(df, colormap='terrain'):
    rgb_colors = colorize_by_value(df['z'].values, colormap)
    df['rgb'] = [uf.pack_rgb_float(r, g, b) for r, g, b in rgb_colors]
    return df

def set_pcd_color_by_z(input_path, output_path, colormap='terrain'):
    pcd_data = read_pcd_file(input_path)
    if pcd_data is None or len(pcd_data) == 0:
        logger.error("input pcd data is empty or could not be loaded. aborting.")
        return
    df = pd.DataFrame(pcd_data)
    df = colorize_by_z(df, colormap)
    save_df_as_pcd(df, output_path)

# from verify_para.py, changed a little bit, need to use this in verify_para.py
def read_pcd_file(filepath, required_fields = None):
    """
    a robust function to read a .pcd file (both ascii and binary formats)
    into a numpy array, returning only the x, y, z columns.
    """
    logger.info(f"reading pcd file: {filepath}")
    
    # --- Step 1: Read and parse the header ---
    # always open in binary mode first to read the header safely
    with open(filepath, 'rb') as f:
        header_lines = []
        data_type = ''
        data_start_position = 0

        while True:
            try:
                line = f.readline().decode('ascii').strip()
                header_lines.append(line)
                if line.startswith('DATA'):
                    data_type = line.split(' ')[1]
                    # record where the binary data will start
                    data_start_position = f.tell() 
                    break
            except Exception as e:
                logger.error(f"error reading pcd header: {e}")
                return None

        # parse header fields
        fields, sizes, types = [], [], []
        for line in header_lines:
            if line.startswith('FIELDS'):
                fields = line.split(' ')[1:]
            elif line.startswith('SIZE'):
                sizes = [int(s) for s in line.split(' ')[1:]]
            elif line.startswith('TYPE'):
                types = line.split(' ')[1:]
        
        if not all([fields, sizes, types, data_type]):
            logger.error("failed to parse complete pcd header (fields, size, type, or data).")
            return None

    # --- Step 2: Read data based on the format type ---
    try:
        if data_type == 'ascii':
            # for ascii, we can use the simple np.loadtxt
            logger.info("detected ascii pcd format.")
            # np.loadtxt can take the filename directly and skip header rows
            # we need to find how many lines the header has
            num_header_lines = len(header_lines)
            points_data = np.loadtxt(filepath, skiprows=num_header_lines)
            
            # create a dictionary to map field names to column indices
            field_to_idx = {name: i for i, name in enumerate(fields)}

        elif data_type == 'binary':
            logger.info("detected binary pcd format.")
            # for binary, we need to construct the dtype and use frombuffer
            type_mapping = {'F': 'f', 'I': 'i', 'U': 'u'}
            dtype_list = []
            for name, size, pcd_type in zip(fields, sizes, types):
                if pcd_type not in type_mapping:
                    logger.error(f"unsupported pcd type '{pcd_type}' for field '{name}'")
                    return None
                numpy_type = type_mapping[pcd_type]
                dtype_list.append((name, f'<{numpy_type}{size}'))
            dtype = np.dtype(dtype_list)

            # read the rest of the file from where the header ended
            with open(filepath, 'rb') as f:
                f.seek(data_start_position)
                binary_data = f.read()

            item_size = dtype.itemsize
            if len(binary_data) % item_size != 0:
                logger.error(f"buffer size ({len(binary_data)}) is not a multiple of element size ({item_size}). file may be corrupt.")
                return None
            
            # this now becomes a structured array
            points_data = np.frombuffer(binary_data, dtype=dtype)
            
            # create a way to access columns by name
            field_to_idx = points_data.dtype.names

        elif data_type == 'binary_compressed':
            # CloudCompare edited and saved pcd, cc might change field order, head -n 11 /mnt/e/edit/output0902v2_ts_1042.pcd
            logger.info("detected binary_compressed pcd format.")

            try:
                pcd = o3d.io.read_point_cloud(filepath)
                points = np.asarray(pcd.points)

                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    colors_255 = (colors * 255).astype(np.uint8)
                    rgb_floats = np.array([uf.pack_rgb_float(r, g, b) for r, g, b in colors_255], dtype=np.float32)
                else:
                    rgb_floats = np.full(len(points), uf.pack_rgb_float(255, 255, 255), dtype=np.float32)
                
                dtype = np.dtype([
                    ('x', 'f4'),
                    ('y', 'f4'), 
                    ('z', 'f4'),
                    ('rgb', 'f4')
                ])
                
                points_data = np.empty(len(points), dtype=dtype)
                points_data['x'] = points[:, 0]
                points_data['y'] = points[:, 1] 
                points_data['z'] = points[:, 2]
                points_data['rgb'] = rgb_floats

                field_to_idx = points_data.dtype.names
                
                logger.info(f"open3d read {len(points)} points")
            except Exception as e:
                logger.error(f"open3d reading error: {e}")
        else:
            logger.error(f"unsupported data type '{data_type}' in pcd file.")
            return None

    except Exception as e:
        logger.error(f"failed to load data from pcd file: {e}", exc_info=True)
        return None

    # --- Step 3: Extract x, y, z and return as a simple (N, 3) array ---
    if required_fields is None:
        return points_data

    # todo: finish
    required_fields = ['x', 'y', 'z']
    if not all(field in field_to_idx for field in required_fields):
        logger.error(f"pcd file is missing one of the required fields: {required_fields}")
        return None
        
    if isinstance(points_data, np.ndarray) and not points_data.dtype.names: # ascii case
        # for ascii, we access by column index
        x_col = points_data[:, field_to_idx['x']]
        y_col = points_data[:, field_to_idx['y']]
        z_col = points_data[:, field_to_idx['z']]
    else: # binary (structured array) case
        x_col = points_data['x']
        y_col = points_data['y']
        z_col = points_data['z']

    return np.stack([x_col, y_col, z_col], axis=-1)


def _load_pcd_data(filepath):
    """
    loads point cloud data from a pcd file (ascii or binary).
    it specifically looks for 'x', 'y', 'z', and 'rgb' fields.

    :param filepath: path to the .pcd file.
    :return: a structured numpy array containing the point data, or none on failure.
    """
    logger.info(f"loading pcd file: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            header = {}
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('DATA'):
                    header['DATA'] = line.split(' ')[1].lower()
                    break
                
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    key, value = parts
                    if key in ['FIELDS', 'TYPE']:
                        header[key] = value.split(' ')
                    elif key == 'SIZE':
                        header[key] = [int(v) for v in value.split(' ')]
                    elif key in ['WIDTH', 'POINTS']:
                        header[key] = int(value)
            
            # check for required fields
            if not all(k in header for k in ['FIELDS', 'SIZE', 'POINTS', 'DATA']):
                logger.error("pcd header is incomplete or malformed.")
                return None
            
            # create the numpy dtype for reading the data
            dtype_list = []
            for i, field in enumerate(header['FIELDS']):
                if field in ['x', 'y', 'z', 'rgb']:
                    # assuming type 'F' (float) and size '4' (32-bit)
                    dtype_list.append((field, '<f4'))
                else:
                    # add a placeholder to skip other fields
                    dtype_list.append((f'_padding_{field}', f'V{header["SIZE"][i]}'))

            structured_dtype = np.dtype(dtype_list)
            
            num_points = header['POINTS']
            
            # read the data block
            if header['DATA'] == 'binary':
                data = np.fromfile(f, dtype=structured_dtype, count=num_points)
            elif header['DATA'] == 'ascii':
                # note: ascii loading is slower and less robust for mixed types
                data_str = f.read().decode('ascii')
                data_lines = data_str.strip().split('\n')
                data_rows = [list(map(float, line.split(' '))) for line in data_lines]
                # convert to structured array to match binary path
                # this assumes the columns in ascii match the expected fields in order
                pddf = pd.DataFrame(data_rows, columns=header['FIELDS'])
                data = np.core.records.fromarrays(pddf[['x', 'y', 'z', 'rgb']].to_numpy().T, 
                                                  names=['x','y','z','rgb'])
            else:
                logger.error(f"unsupported data type '{header['DATA']}' in pcd file.")
                return None

            # select only the fields we care about for the output
            final_data = data[['x', 'y', 'z', 'rgb']]
            logger.info(f"successfully loaded {len(final_data)} points.")
            return final_data

    except Exception as e:
        logger.error(f"failed to load pcd file: {e}", exc_info=True)
        return None


def write_pcd_binary(filepath, points_data, has_timestamp=False, viewpoint=[0, 0, 0, 1, 0, 0, 0]):
    """
    writes a structured numpy array to a binary .pcd file format. 
    Note without rgb, ccviewer will render it white, might not be seen if the background is white.

    :param filepath: path for the output .pcd file.
    :param points_data: structured numpy array with fields 'x', 'y', 'z', 'rgb'.
    :param viewpoint: [tx,ty,tz,qw,qx,qy,qz]
    """
    num_points = len(points_data)
    logger.info(f"writing {num_points} points to binary pcd file: {filepath}")
    
    if has_timestamp:
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb timestamp
SIZE 4 4 4 4 8
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT {' '.join(map(str, viewpoint))}
POINTS {num_points}
DATA binary
"""
    else:
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT {' '.join(map(str, viewpoint))}
POINTS {num_points}
DATA binary
"""

    try:
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(points_data.tobytes())
        logger.info("file writing complete.")
    except Exception as e:
        logger.error(f"failed to write pcd file: {e}", exc_info=True)


def downsample_pcd_with_voxel_grid(input_path, output_path, voxel_size):
    """
    loads a point cloud, downsamples it using a voxel grid filter while
    preserving color, and saves the result as a new binary pcd file.

    :param input_path: path to the source .pcd file.
    :param output_path: path to save the downsampled .pcd file.
    :param voxel_size: the side length of the cubic voxel (e.g., 0.1 for 10cm).
    """
    # 1. load the initial point cloud data
    pcd_data = read_pcd_file(input_path)
    if pcd_data is None or len(pcd_data) == 0:
        logger.error("input pcd data is empty or could not be loaded. aborting.")
        return

    # 2. perform voxel grid downsampling
    logger.info(f"starting voxel downsampling with voxel size: {voxel_size}...")
    logger.info(f"points before downsampling: {len(pcd_data)}")

    # use pandas for efficient grouping and aggregation
    df = pd.DataFrame(pcd_data)

    # compute discrete voxel indices for each point
    df['vx'] = (df['x'] / voxel_size).astype(int)
    df['vy'] = (df['y'] / voxel_size).astype(int)
    df['vz'] = (df['z'] / voxel_size).astype(int)

    # strategy 1: use mean
    # group by voxel index and aggregate
    # - xyz coordinates are averaged to get the voxel centroid.
    # - rgb color is taken from the first point in the voxel to avoid averaging colors.
    # voxel_groups = df.groupby(['vx', 'vy', 'vz'])
    # downsampled_df = voxel_groups.agg({
    #     'x': 'mean',
    #     'y': 'mean',
    #     'z': 'mean',
    #     'rgb': 'first'
    # }).reset_index()

    # strategy 2, use voxel center
    # select color
    white_rgb_float = 0
    if df['rgb'].dtype == np.float32:
        white_rgb_float = uf.pack_rgb_float(255, 255, 255)
    elif df['rgb'].dtype == np.uint32:
        white_rgb_float = uf.pack_rgb_uint32(255, 255, 255)
    else:
        raise ValueError("dtype not supported: {}".format(df['rgb'].dtype))

    # extremely slow
    # def select_color(rgb_series):
    #     # if exist non-white, use non-white, otherwise use white
    #     non_white_rgbs = rgb_series[rgb_series != white_rgb_float]
    #     if not non_white_rgbs.empty:
    #         return non_white_rgbs.iloc[0]
    #     else:
    #         return white_rgb_float
    # downsampled_df = df.groupby(['vx', 'vy', 'vz']).agg(rgb=('rgb', select_color)).reset_index()





    




    # white_mask = df['rgb'] == white_rgb_float
    # non_white_df = df[~white_mask]
    # white_df = df[white_mask]
    
    # # 对非白色记录分组，取第一个
    # if not non_white_df.empty:
    #     non_white_grouped = non_white_df.groupby(['vx', 'vy', 'vz']).first().reset_index()
    # else:
    #     non_white_grouped = pd.DataFrame(columns=['vx', 'vy', 'vz', 'rgb'])
    
    # # 对白色记录分组，但只保留在non_white_grouped中不存在的坐标
    # if not white_df.empty:
    #     white_grouped = white_df.groupby(['vx', 'vy', 'vz']).first().reset_index()
    #     # 找出non_white_grouped中没有的坐标
    #     merged = white_grouped.merge(
    #         non_white_grouped[['vx', 'vy', 'vz']], 
    #         on=['vx', 'vy', 'vz'], 
    #         how='left', 
    #         indicator=True
    #     )
    #     white_only = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
    # else:
    #     white_only = pd.DataFrame(columns=['vx', 'vy', 'vz', 'rgb'])
    
    # # 合并结果
    # downsampled_df = pd.concat([non_white_grouped, white_only], ignore_index=True)







    downsampled_df = df.groupby(['vx', 'vy', 'vz']).first().reset_index()
    downsampled_df['x'] = (downsampled_df['vx'] + 0.5) * voxel_size
    downsampled_df['y'] = (downsampled_df['vy'] + 0.5) * voxel_size
    downsampled_df['z'] = (downsampled_df['vz'] + 0.5) * voxel_size

    logger.info(f"points after downsampling: {len(downsampled_df)}")
    save_df_as_pcd(downsampled_df[['x', 'y', 'z', 'rgb']], output_path)




DUMMY_FIELD_PREFIX = '_'

type_mappings = {
    PointField.INT8: ('I', 1, 'i1'),
    PointField.UINT8: ('U', 1, 'u1'), 
    PointField.INT16: ('I', 2, 'i2'),
    PointField.UINT16: ('U', 2, 'u2'),
    PointField.INT32: ('I', 4, 'i4'),
    PointField.UINT32: ('U', 4, 'u4'),
    PointField.FLOAT32: ('F', 4, 'f4'),
    PointField.FLOAT64: ('F', 8, 'f8'),
}

mapping = {
    'float32': PointField.FLOAT32,
    'float64': PointField.FLOAT64,
    'int32': PointField.INT32,
    'uint32': PointField.UINT32,
    'int16': PointField.INT16,
    'uint16': PointField.UINT16,
    'int8': PointField.INT8,
    'uint8': PointField.UINT8,
}

def _generate_pcd_header(fields, num_points, data_format):
    field_names = []
    sizes = []
    types = []
    counts = []
    
    for field in fields:
        field_names.append(field.name)
        pcd_type, pcd_size, _ = type_mappings[field.datatype]
        types.append(pcd_type)
        sizes.append(str(pcd_size))
        counts.append(str(field.count))
    
    header_lines = [
        '# .PCD v.7 - Point Cloud Data file format',
        'VERSION .7',
        f"FIELDS {' '.join(field_names)}",
        f"SIZE {' '.join(sizes)}",
        f"TYPE {' '.join(types)}",
        f"COUNT {' '.join(counts)}",
        f"WIDTH {num_points}",
        f"HEIGHT 1",
        'VIEWPOINT 0 0 0 1 0 0 0',
        f"POINTS {num_points}",
        f"DATA {data_format.lower()}"
    ]
    
    return header_lines

def _write_ascii_pcd_fast(filepath, header_lines, points_array):
    with open(filepath, 'w') as f:
        f.write('\n'.join(header_lines) + '\n')
        np.savetxt(f, points_array, fmt='%.6g', delimiter=' ')

def _write_binary_pcd_fast(filepath, header_lines, points_array, fields):
    """now points_array is float64, will be converted to correct type"""
    with open(filepath, 'wb') as f:
        header_str = '\n'.join(header_lines) + '\n'
        f.write(header_str.encode('utf-8'))
        
        format_chars = []
        for field in fields:
            if field.datatype == PointField.FLOAT32:
                format_chars.append('f')
            elif field.datatype == PointField.FLOAT64:
                format_chars.append('d')
            elif field.datatype == PointField.UINT8:
                format_chars.append('B')
            elif field.datatype == PointField.INT8:
                format_chars.append('b')
            elif field.datatype == PointField.UINT16:
                format_chars.append('H')
            elif field.datatype == PointField.INT16:
                format_chars.append('h')
            elif field.datatype == PointField.UINT32:
                format_chars.append('I')
            elif field.datatype == PointField.INT32:
                format_chars.append('i')
        format_str = '<' + ''.join(format_chars)

        has_int_types = bool(set(format_chars) & {'I', 'H', 'B', 'i', 'h', 'b'})
        if not has_int_types:
            for point in points_array:
                f.write(struct.pack(format_str, *point))
            return
        
        # slow
        for point in points_array:
            converted_point = []
            for value, fmt_char in zip(point, format_chars):
                if fmt_char in 'IHBihb':
                    converted_point.append(int(value))
                else:  # 'fd'
                    converted_point.append(float(value))
            f.write(struct.pack(format_str, *converted_point))

def save_df_as_pcd(df, filepath, data_format='binary'):
    points_data = df.to_numpy()  # float64, not reserving dtype from df

    offset = 0
    fields = []
    for col in df.columns:
        datatype = mapping.get(str(df[col].dtype), None)
        size = type_mappings[datatype][1]
        fields.append(PointField(name=col, offset=offset, datatype=datatype, count=1))
        offset += size
    
    header_lines = _generate_pcd_header(fields, len(points_data), data_format)

    if data_format.lower() == 'binary':
        _write_binary_pcd_fast(filepath, header_lines, points_data, fields)
    else:
        _write_ascii_pcd_fast(filepath, header_lines, points_data)

class PointCloudSaverFast:
    def save_df_as_pcd(self, df, filepath, data_format='binary'):
        points_data = df.to_numpy()

        fields = [PointField(name=col, offset=i*4, 
                 datatype=PointField.FLOAT32, count=1) for i, col in enumerate(df.columns)]
        
        header_lines = self._generate_pcd_header(fields, len(points_data), data_format)

        if data_format.lower() == 'binary':
            _write_binary_pcd_fast(filepath, header_lines, points_data, fields)
        else:
            self._write_ascii_pcd_fast(filepath, header_lines, points_data)

    def msg_as_df(self, msg, field_names=None, skip_nans=True):
        """
        msg: sensor_msgs.msg.PointCloud2 message
        field_names: None or list
        """
        try:
            selected_fields, field_indices = self._get_selected_fields(msg.fields, field_names)
            
            if not selected_fields:
                logger.warning("No valid fields found")
                return
            
            points_data = self._parse_pointcloud_fast(msg, selected_fields, field_indices, skip_nans)

            if len(points_data) == 0:
                logger.warning("no valid points found in the point cloud")
                return pd.DataFrame()
            
            df = pd.DataFrame(points_data, columns=[field.name for field in selected_fields])
            return df
        
        except Exception as e:
            logger.error(f'Failed to save pointcloud: {e}')
            raise
    
    def save_msg_as_pointcloud(self, msg, filepath, data_format='binary', field_names=None, skip_nans=True):
        """
        msg: sensor_msgs.msg.PointCloud2 message
        data_format: 'ascii' or 'binary' 
        field_names: None or list
        """
        try:
            selected_fields, field_indices = self._get_selected_fields(msg.fields, field_names)
            
            if not selected_fields:
                logger.warning("No valid fields found")
                return
            
            points_data = self._parse_pointcloud_fast(msg, selected_fields, field_indices, skip_nans)
            
            if len(points_data) == 0:
                logger.warning("No valid points found in the point cloud")
                return
            
            header_lines = self._generate_pcd_header(selected_fields, len(points_data), data_format)

            if data_format.lower() == 'binary':
                _write_binary_pcd_fast(filepath, header_lines, points_data, selected_fields)
            else:
                self._write_ascii_pcd_fast(filepath, header_lines, points_data)
                
            logger.info(f'Successfully saved {len(points_data)} points to {filepath} in {data_format} format')
            
        except Exception as e:
            logger.error(f'Failed to save pointcloud: {e}')
            raise
    
    def _get_selected_fields(self, all_fields, field_names):
        valid_fields = [field for field in all_fields 
                       if not field.name.startswith(DUMMY_FIELD_PREFIX)]
        
        if field_names is None:
            selected_fields = valid_fields
            field_indices = list(range(len(valid_fields)))
        else:
            selected_fields = []
            field_indices = []
            field_name_to_field = {field.name: field for field in valid_fields}
            
            for name in field_names:
                if name in field_name_to_field:
                    selected_fields.append(field_name_to_field[name])
                    for i, field in enumerate(valid_fields):
                        if field.name == name:
                            field_indices.append(i)
                            break
        
        return selected_fields, field_indices
    
    def _parse_pointcloud_fast(self, msg, selected_fields, field_indices, skip_nans):
        num_points = msg.width * msg.height
        
        dtype_list = []
        for field in msg.fields:
            if field.name.startswith(DUMMY_FIELD_PREFIX):
                continue
            _, _, np_type = type_mappings.get(field.datatype, ('F', 4, 'f4'))
            dtype_list.append((field.name, np_type))
        
        byte_order = '<' if not msg.is_bigendian else '>'
        dtype = np.dtype(dtype_list)
        if byte_order == '>':
            dtype = dtype.newbyteorder('>')
        
        try:
            points_structured = np.frombuffer(msg.data, dtype=dtype)
            if len(points_structured) != num_points:
                points_structured = self._parse_with_stride(msg, dtype, num_points)
        except ValueError:
            points_structured = self._parse_with_stride(msg, dtype, num_points)
        
        selected_data = []
        for i, field in enumerate(selected_fields):
            selected_data.append(points_structured[field.name])
        
        # (num_points, num_fields)
        points_array = np.column_stack(selected_data)
        
        if skip_nans:
            valid_mask = ~np.isnan(points_array).any(axis=1)
            points_array = points_array[valid_mask]
        
        return points_array
    
    def _parse_with_stride(self, msg, dtype, num_points):
        """使用stride手动解析数据（处理padding情况）"""
        point_step = msg.point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # 创建结构化数组
        points_list = []
        for i in range(num_points):
            start_idx = i * point_step
            point_data = data[start_idx:start_idx + dtype.itemsize]
            if len(point_data) == dtype.itemsize:
                point = np.frombuffer(point_data, dtype=dtype)[0]
                points_list.append(point)
        
        return np.array(points_list, dtype=dtype)
    
    def _generate_pcd_header(self, fields, num_points, data_format):
        field_names = []
        sizes = []
        types = []
        counts = []
        
        for field in fields:
            field_names.append(field.name)
            pcd_type, pcd_size, _ = type_mappings[field.datatype]
            types.append(pcd_type)
            sizes.append(str(pcd_size))
            counts.append(str(field.count))
        
        header_lines = [
            '# .PCD v.7 - Point Cloud Data file format',
            'VERSION .7',
            f"FIELDS {' '.join(field_names)}",
            f"SIZE {' '.join(sizes)}",
            f"TYPE {' '.join(types)}",
            f"COUNT {' '.join(counts)}",
            f"WIDTH {num_points}",
            f"HEIGHT 1",
            'VIEWPOINT 0 0 0 1 0 0 0',
            f"POINTS {num_points}",
            f"DATA {data_format.lower()}"
        ]
        
        return header_lines
    
    def _write_ascii_pcd_fast(self, filepath, header_lines, points_array):
        with open(filepath, 'w') as f:
            f.write('\n'.join(header_lines) + '\n')
            np.savetxt(f, points_array, fmt='%.6g', delimiter=' ')
    


if __name__ == '__main__':
    # --- configuration ---
    # set the input file, output file, and voxel size parameter here.
    # input_pcd_file = "/mnt/e/edit/output0902v2_1042_global.pcd"
    # output_pcd_file = "/mnt/e/edit/output0902v2_1042_global_downsampled.pcd"
    
    # # default to 0.2, 0.1 for visualization, 0.3-0.5 for matching
    # voxel_leaf_size = 0.1

    # if not os.path.exists(input_pcd_file):
    #     logger.error(f"input file not found: {input_pcd_file}")
    #     sys.exit(1)
        
    # downsample_pcd_with_voxel_grid(
    #     input_path=input_pcd_file,
    #     output_path=output_pcd_file,
    #     voxel_size=voxel_leaf_size
    # )

    # _load_pcd_data('/home/weizh/data/output0831_testpy.pcd')
    # read_pcd_file('/home/weizh/data/output0831_testpy.pcd')

    # saver = PointCloudSaverFast()
    # saver.save_msg_as_pointcloud(pcd_msg, "output.pcd", data_format='ascii', field_names=["x", "y", "z"])
    # saver.save_msg_as_pointcloud(pcd_msg, "output_binary.pcd", data_format='binary', field_names=["x", "y", "z", "intensity"])
    # saver.save_msg_as_pointcloud(pcd_msg, "output_all_fields.pcd", data_format='ascii')  # 输出所有字段



    # set_pcd_color_by_z("/mnt/e/downloads/output0902v2_1042_global_downsampled.pcd", "/mnt/e/downloads/output0902v2_1042_global_downsampled_c.pcd", 'rviz')
    # set_pcd_color_by_z("/home/weizh/Downloads/LOAM/GlobalMap.pcd", "/home/weizh/Downloads/LOAM/GlobalMap_c.pcd", 'rainbow')
    set_pcd_color_by_z("/home/weizh/data/output0927_ds.pcd", "/home/weizh/data/output0927_ds_rviz.pcd", 'viridis')
