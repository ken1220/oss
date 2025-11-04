# app.py

from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

# ★★★★★ ここにOpenCVを使った画像処理ロジックを定義 ★★★★★
def process_card_image(img_data):
    # 画像データのデコード
    npimg = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return None, "画像のデコードに失敗しました。"

    result_image = image.copy() 

    # 輪郭検出処理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # 四角形の輪郭を検出し、描画
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(result_image, [screenCnt], -1, (0, 255, 0), 5) 
            break

    message = "輪郭を検出しました！" if screenCnt is not None else "四角形の輪郭が見つかりませんでした。"
    
    return result_image, message
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"message": "画像がありません。"}), 400
    
    file = request.files['image']
    img_data = file.read()
    
    processed_img, message = process_card_image(img_data)
    
    if processed_img is None:
        return jsonify({"message": message}), 500

    # 処理結果の画像をJPEG形式でメモリに保存
    is_success, buffer = cv2.imencode(".jpg", processed_img)
    if not is_success:
        return jsonify({"message": "処理結果のエンコードに失敗しました。"}), 500

    # 画像をレスポンスとして返す
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/jpeg'
    )

if __name__ == '__main__':
    # ローカルテスト用
    app.run(host='0.0.0.0', port=5000)
