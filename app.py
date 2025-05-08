import os
from flask import Flask, render_template, request, redirect, url_for, jsonify

from ml.config import RESIZE_VALUE, DATA_PATH
from ml.model import load_model
from ml.data_utils import get_aircraft_types
from ml.api_model import detect_and_draw_aircraft

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Грузим модель и классы один раз при старте
aircraft_types = get_aircraft_types()
model_path = os.path.join(DATA_PATH, "aircraft_model.keras")
model = load_model(model_path)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result_url = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'upload.jpg')
            out_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
            file.save(img_path)
            detect_and_draw_aircraft(img_path, out_path, model, aircraft_types, resize_value=RESIZE_VALUE)
            result_url = url_for('static', filename='result.jpg')
            # Если запрос AJAX - возвращаем JSON
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'result_url': result_url})
            # Иначе обычный рендер (на случай прямого перехода)
            return render_template('index.html', result_url=result_url)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
