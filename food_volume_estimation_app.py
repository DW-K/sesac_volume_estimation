import os
ROOT_DIR = os.path.abspath(".")
import argparse
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model, model_from_json
from food_volume_estimation.volume_estimator import VolumeEstimator, DensityDatabase
from food_volume_estimation.depth_estimation.custom_modules import *
from food_volume_estimation.food_segmentation.food_segmentator import FoodSegmentator
from flask import Flask, request, jsonify, make_response, abort
import base64
import io
from PIL import Image


app = Flask(__name__)
estimator = None
density_db = None

def load_volume_estimator(depth_model_architecture, depth_model_weights,
        segmentation_model_weights, density_db_source):
    """Loads volume estimator object and sets up its parameters."""
    # Create estimator object and intialize
    global estimator
    estimator = VolumeEstimator(arg_init=False)
    with open(depth_model_architecture, 'r') as read_file:
        custom_losses = Losses()
        objs = {'ProjectionLayer': ProjectionLayer,
                'ReflectionPadding2D': ReflectionPadding2D,
                'InverseDepthNormalization': InverseDepthNormalization,
                'AugmentationLayer': AugmentationLayer,
                'compute_source_loss': custom_losses.compute_source_loss}
        model_architecture_json = json.load(read_file)
        estimator.monovideo = model_from_json(model_architecture_json,
                                              custom_objects=objs)
    estimator._VolumeEstimator__set_weights_trainable(estimator.monovideo,
                                                      False)
    estimator.monovideo.load_weights(depth_model_weights)
    estimator.model_input_shape = (
        estimator.monovideo.inputs[0].shape.as_list()[1:])
    depth_net = estimator.monovideo.get_layer('depth_net')
    estimator.depth_model = Model(inputs=depth_net.inputs,
                                  outputs=depth_net.outputs,
                                  name='depth_model')
    print('[*] Loaded depth estimation model.')

    # Depth model configuration
    MIN_DEPTH = 0.01
    MAX_DEPTH = 10
    estimator.min_disp = 1 / MAX_DEPTH
    estimator.max_disp = 1 / MIN_DEPTH
    estimator.gt_depth_scale = 0.35 # Ground truth expected median depth

    # Create segmentator object
    estimator.segmentator = FoodSegmentator(segmentation_model_weights)
    # Set plate adjustment relaxation parameter
    estimator.relax_param = 0.01

    # Need to define default graph due to Flask multiprocessing
    global graph
    graph = tf.get_default_graph()

    if density_db_source is not None:
        # Load food density database
        global density_db
        density_db = DensityDatabase(density_db_source)

@app.route('/predict', methods=['POST'])
def volume_estimation():
    """Receives an HTTP multipart request and returns the estimated 
    volumes of the foods in the image given.

    Multipart form data:
        img: The image file to estimate the volume in.
        plate_diameter: The expected plate diamater to use for depth scaling.
        If omitted then no plate scaling is applied.

    Returns:
        The array of estimated volumes in JSON format.
    """
    # Decode incoming byte stream to get an image
    if 'img' not in request.files:
        return make_response(jsonify({'error': 'No img part in the request.'}), 400)
    file = request.files['img']

    if file.filename == '':
            return make_response(jsonify({'error': 'No image found.'}), 400)
    # print(f'file: {file}, {type(file)}')
    # file.seek(0)
    # image_data = file.read()
    # print(f'image_data :{image_data} {type(image_data)}')
    # np_img = np.frombuffer(image_data, np.uint8)
    # print(f'np_img :{np_img} {type(np_img)}')
    # img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    # print(f'img :{img} {type(img)}')

    # np_img = np.fromstring(file.read(), np.uint8)
    # np_img = np.frombuffer(file.read(), np.uint8)
    # # np_img = np.fromstring(file.stream.read(), np.uint8)
    # # np_img = np.frombuffer(file.stream.read(), np.uint8)
    # print("np_img: ")
    # print(np_img, type(np_img))
    # img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    try:
        content = request.get_json()
        print(f'content: {content}')
        img_encoded = content['img']
        print(f'img_encoded: {img_encoded}')
        img_byte_string = ' '.join([str(x) for x in img_encoded]) # If in byteArray
        print(f'img_byte_string: {img_byte_string}')
        #img_byte_string = base64.b64decode(img_encoded) # Decode if in base64
        np_img = np.fromstring(img_byte_string, np.int8, sep=' ')
        print(f'np_img: {np_img}')
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        print(f'img: {img}')
    except Exception as e:
        print(e)
        abort(406)

    # Get food type
    try:
        food_type = request.form.get('food', None)
    except Exception as e:
        abort(406)

    # Get expected plate diameter from form data or set to 0 and ignore
    try:
        # plate_diameter = float(content['plate_diameter'])
        plate_diameter = float(request.form.get('plate_diameter', 0))
    except Exception as e:
        print('set plate_diameter 0')
        plate_diameter = 0

    # Estimate volumes
    with graph.as_default():
        volumes = estimator.estimate_volume(img, fov=70,
            plate_diameter_prior=plate_diameter)
    
    try:
    # Convert to mL
        volumes = [v * 1e6 for v in volumes]
    except:
        print("volume has wrong value")
        return make_response(jsonify({'volume': 0}), 200)
    
    # Convert volumes to weight - assuming a single food type
    try:
        db_entry = density_db.query(food_type)
        print(f'db_entry: {db_entry}')
        density = float(db_entry[1])
        weight = 0
        for v in volumes:
            weight += v * density

        # Return values
        return_vals = {
            'food_type_match': db_entry[0],
            'weight': weight
        }
    except Exception as e:
        print("density Error")
        print(e)
        return_vals = {
            'volume(mL)': sum(volumes)
        }
    # return make_response(jsonify(return_vals), 200)
    return make_response(return_vals, 200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Food volume estimation API.')
    parser.add_argument('--depth_model_architecture', type=str,
                        help='Path to depth model architecture (.json).',
                        default=f'{ROOT_DIR}/models/monovideo_fine_tune_food_videos.json',
                        required=False)
    parser.add_argument('--depth_model_weights', type=str,
                        help='Path to depth model weights (.h5).',
                        default=f'{ROOT_DIR}/models/monovideo_fine_tune_food_videos.h5',
                        required=False)
    parser.add_argument('--segmentation_model_weights', type=str,
                        help='Path to segmentation model weights (.h5).',
                        default=f'{ROOT_DIR}/models/mask_rcnn_food_segmentation.h5',
                        required=False)
    parser.add_argument('--density_db_source', type=str,
                        help=('Path to food density database (.xlsx) ' +
                              'or Google Sheets ID.'),
                        default=f'{ROOT_DIR}/database/database.csv',
                        required=False)
    args = parser.parse_args()
    
    load_volume_estimator(depth_model_architecture = args.depth_model_architecture,
                          depth_model_weights= args.depth_model_weights, 
                          segmentation_model_weights = args.segmentation_model_weights,
                          density_db_source = args.density_db_source)
    app.run(host='0.0.0.0', port='8888')

