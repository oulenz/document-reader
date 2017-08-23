import cv2
import datetime
import json
import logging
import numpy as np
import os
import socket

from flask import Flask, request, Response, jsonify
from uuid import uuid4

from document_scanner.document_scanner import Document_scanner
from document_scanner.os_wrapper import BASE_DIR_PATH

CONFIG_FILE_PATH = os.path.join(BASE_DIR_PATH, 'data', 'paths.txt')

# Logger initialisation. This must happen before any calls to debug(), info(), etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# In each file/module, do this to get the module name in the logs
logger = logging.getLogger(__name__)

# API initialisation
app = Flask(__name__)

# Document scanner initialisation (loads models, analises template)
scanner = Document_scanner(CONFIG_FILE_PATH)


def to_np_array(image_file):
    return cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)


@app.route('/')
def home():
    logging.info('/, hostname: ' + str(socket.gethostname()))

    return jsonify(
        health='OK!',
        hostname=str(socket.gethostname())
    )


@app.route('/predict', methods=['POST'])
def predict():
    logger.info('/predict, hostname: ' + str(socket.gethostname()))

    if 'image' not in request.files:
        logger.info('Missing image parameter')
        return Response('Missing image parameter', 400)

    if 'caseId' not in request.form:
        logger.info('Missing caseId parameter')
        return Response('Missing caseId parameter', 400)

    request_id = str(datetime.datetime.now()) + '-' + uuid4().hex
    request_path = os.path.join(os.path.dirname(__file__), 'storage', 'claim_document', request_id)

    if not os.path.exists(request_path):
        os.makedirs(request_path)

    # Store the request data
    parameter_path = os.path.join(request_path, 'request.txt')
    with open(parameter_path, 'w') as f:
        for key in request.form:
            f.write(key + ': ' + request.form[key] + '\n')

    image = to_np_array(request.files['image'])

    original_path = os.path.join(request_path, 'original.jpg')
    cv2.imwrite(original_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    document = scanner.develop_document(original_path)

    resized_path = os.path.join(request_path, 'resized.jpg')
    cv2.imwrite(resized_path, document.resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    if not document.can_create_scan():
        logger.info('No match with template')
        return Response('No match with template', 400)

    scan_path = os.path.join(request_path, 'scan.jpg')
    cv2.imwrite(scan_path, document.scan, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    content_json = document.get_content_labels_json()
    content = document.get_content_labels()

    content_path = os.path.join(request_path, 'content.json')

    with open(content_path, 'w') as outfile:
        json.dump(content_json, outfile)

    logger.info('Content extracted by claimform endpoint: ' + content_json + ' : ' + str(request_path))

    return jsonify(
        **content,
        hostname=str(socket.gethostname())
    )


@app.errorhandler(500)
def server_error(e):
    app.logger.error(str(e))
    response = Response('An internal error occurred. ' + str(e), 500)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')
