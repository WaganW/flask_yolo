from flask import Flask, request, jsonify, render_template, send_file, flash
from flask import send_from_directory
import torch
import os
import shutil
from ultralytics import YOLO

app = Flask(__name__)

# 定义全局变量
filename = None
file_path = None

@app.route('/', methods=['GET', 'POST'])
def upload():
    global filename
    global file_path
    if request.method == 'POST':  # POST是一种请求方式
        # 从表单中获取上传的文件
        f = request.files['file']  # request.files 函数作用就是获取前端名为 'file' 的文件信息
        filename = f.filename  # 获取前端上传图片名字
        file_path = os.path.join(os.getcwd(), filename)  # 本地路径+图片名字= 文件路径（file-path）

        f.save(file_path)  # 保存上传的图片到本地目录下，方便后续推理，直接找到图片

        model_path = request.form.get('./best1.pt', './best2.pt')
        model = YOLO(model_path)
        source = file_path
        model.predict(source, save=True, imgsz=640)

        # 定义原始图片目标路径
        original_image_path1 = 'runs/detect/predict/' + str(filename) 
        original_image_path2 = 'runs/detect/predict'
        destination_path = 'result_pic'
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        shutil.move(original_image_path1, os.path.join(destination_path, filename))
        if os.path.exists(original_image_path2):
            shutil.rmtree(original_image_path2)

    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global filename
    global file_path
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(os.getcwd(), filename)
        file.save(file_path)
        model_path = request.form.get('./best2.pt', './best1.pt')#谁放前面拿谁
        model = YOLO(model_path)
        source = file_path
        model.predict(source, save=True, imgsz=640)
        original_image_path1 = 'runs/detect/predict/' + str(filename) 
        original_image_path2 = 'runs/detect/predict'
        destination_path = 'result_pic'
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        shutil.move(original_image_path1, os.path.join(destination_path, filename))
        if os.path.exists(original_image_path2):
            shutil.rmtree(original_image_path2)
        return jsonify({'detected_image': '/sh/' + filename}), 200

@app.route('/sh/<filename>', methods=['GET'])
def send_js(filename):
    return send_from_directory('result_pic', filename)

if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host='0.0.0.0', port=5000)