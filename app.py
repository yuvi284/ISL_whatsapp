from flask import Flask, Response, logging, render_template, request, redirect, url_for, session, flash, jsonify
from db_helper import DBHelper
import mediapipe as mp
import bcrypt
from werkzeug.utils import secure_filename
import os
import threading
import cv2
import numpy as np
from flask import Response
from collections import deque
import tensorflow as tf
import mediapipe as mp
from scipy.stats import entropy
import keras
from msg2isl import TranslationHandler, TextProcessor, DatabaseHandler  # If using as module
from compress import compress_sentence
# from googletrans import Translator
from collections import defaultdict
from MLPinferece import HandGestureRecognition  # ensure import

app = Flask(__name__)
app.secret_key = 'rahul'  # Change this to a random secret key
db_helper = DBHelper()
final_sentence = []

camera_active = False
camera_thread = None
output_frame = None
lock = threading.Lock()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'login':
            mobile = request.form['mobile']
            password = request.form['password']

            user = db_helper.get_user_by_mobile(mobile)

            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                session['user_info'] = {
                    "mobile": user['mobile'],
                    "username": user['username'],
                    "email": user.get('email'),
                    "gender": user.get('gender'),
                    "profile_picture": '/static' + user.get('profile_picture').split('static')[-1].replace('\\', '/') if user.get('profile_picture') else None
                }
                session['mobile'] = user['mobile']  # Save separately for easy access
                return redirect('/teacher_dashboard')

            else:
                flash('Invalid mobile or password.', 'error')

        elif action == 'register':
            mobile = request.form['mobile']
            username = request.form['username']
            password = request.form['password']
            email = request.form.get('email')
            gender = request.form.get('gender')

            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            file = request.files.get('profile_picture')
            profile_pic_path = None
            if file and file.filename != "":
                from werkzeug.utils import secure_filename
                filename = secure_filename(f"{mobile}_{file.filename}")
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            if db_helper.user_exists(mobile):
                flash('User with this mobile already exists.', 'error')
            else:
                db_helper.register_user(mobile, username, email, gender, hashed_pw, profile_pic_path)
                flash('Registration successful. Please log in.', 'success')
                return redirect(url_for('login'))

    return render_template('login_register.html', action='login')

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
# UPLOAD_FOLDER = '/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        mobile = request.form['mobile']
        username = request.form['username']
        email = request.form.get('email')
        gender = request.form.get('gender')
        password = request.form['password']
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        profile_pic_path = None
        file = request.files.get('profile_picture')
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{mobile}_{file.filename}")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            profile_pic_path = '/static' +profile_pic_path.split('static')[-1].replace('\\', '/') if profile_pic_path else None
        success = db_helper.register_user(
            mobile, username, email, gender, hashed_pw, profile_pic_path
        )
        if success:
            return redirect('/login')  # Redirect after successful registration
        else:
            return "Registration failed. User may already exist."

    return render_template('new_registeration.html')

@app.route('/teacher_dashboard')
def teacher_dashboard():
    if 'user_info' not in session:
        return redirect('/login')

    current_mobile = session.get('mobile')

    saved_contacts = db_helper.GetSavedContacts(current_mobile)
    print(saved_contacts, "Saved Contacts")

    return render_template('teacher_dashboard.html',
                           user_info=session['user_info'],
                           students=saved_contacts)


@app.route('/search_contact')
def search_contact():
    search_mobile = request.args.get('mobile')
    owner_mobile = session.get('mobile')

    if not search_mobile or not owner_mobile:
        return jsonify({"status": "error"})

    result = db_helper.SearchContact(owner_mobile, search_mobile)
    return jsonify(result)

@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    sender_mobile = session.get("mobile")
    receiver_mobile = data.get("receiver_mobile")
    message_content = data.get("message")

    if not sender_mobile or not receiver_mobile or not message_content:
        return jsonify({"success": False, "error": "Missing fields"}), 400

    success = db_helper.SaveChatMessage(sender_mobile, receiver_mobile, message_content)
    return jsonify({"success": success})


@app.route('/save_contact', methods=['POST'])
def save_contact():
    data = request.get_json()
    contact_mobile = data.get('contact_mobile')
    nickname = data.get('nickname')
    owner_mobile = session.get('mobile')

    if not contact_mobile or not nickname or not owner_mobile:
        return jsonify({"success": False})

    success = db_helper.SaveContact(owner_mobile, contact_mobile, nickname)
    return jsonify({"success": success})

@app.route('/get_saved_contacts')
def get_saved_contacts():
    owner_mobile = session.get('mobile')
    if not owner_mobile:
        return jsonify({"contacts": []})

    contacts = db_helper.GetSavedContacts1(owner_mobile)
    return jsonify({"contacts": contacts})


@app.route('/logout')
def logout():
    session.clear()  # Clears all session data
    return redirect(url_for('login'))  # Redirect to login or home

@app.route('/get_messages')
def get_messages():
    contact = request.args.get('contact')
    user_mobile = session.get('mobile')

    if not contact or not user_mobile:
        return jsonify({'messages': []})

    messages = db_helper.GetChatMessages(user_mobile, contact)
    return jsonify({'messages': messages})


@app.route('/get_contact_image')
def get_contact_image():
    mobile = request.args.get('mobile')

    if not mobile:
        return jsonify({'profile_picture': '/static/uploads/blank.png'})

    image_path = db_helper.get_profile_picture(mobile)

    if image_path:
        # Extract the relative static path
        if 'static' in image_path:
            relative_path = '/static' + image_path.split('static')[-1].replace('\\', '/')
        else:
            # fallback if something goes wrong
            relative_path = '/static/uploads/blank.png'
    else:
        relative_path = '/static/uploads/blank.png'

    return jsonify({'profile_picture': relative_path})


@app.route('/mark_seen', methods=['POST'])
def mark_seen():
    data = request.get_json()
    user_mobile = session['user_info']['mobile']
    contact_mobile = data.get('contact_mobile')

    if not user_mobile or not contact_mobile:
        return {"success": False}

    db_helper.mark_messages_as_seen(contact_mobile, user_mobile)
    return {"success": True}

@app.route('/delete_message', methods=['POST'])
def delete_message():
    data = request.get_json()
    message_id = data.get('message_id')
    print(message_id," messageID")
    success = db_helper.DeleteMessage(message_id)

    return jsonify({"success": success})


@app.route('/edit_contact_name', methods=['POST'])
def edit_contact_name():
    data = request.get_json()
    mobile = data.get('mobile')
    new_name = data.get('new_name')
    user_mobile = session.get('mobile')

    if not mobile or not new_name:
        return jsonify(success=False)

    result = db_helper.update_contact_name(user_mobile, mobile, new_name)
    return jsonify(success=result)

@app.route('/delete_contact', methods=['POST'])
def delete_contact():
    data = request.get_json()
    mobile = data.get('mobile')
    user_mobile = session.get('mobile')

    if not mobile:
        return jsonify(success=False)

    result = db_helper.delete_contact(user_mobile, mobile)
    return jsonify(success=result)

@app.route('/delete_contact_messages', methods=['POST'])
def delete_contact_messages():
    data = request.get_json()
    mobile = data.get('mobile')
    user_mobile = session.get('mobile')

    if not mobile:
        return jsonify(success=False)

    result = db_helper.delete_contact_messages(user_mobile, mobile)
    return jsonify(success=result)

@app.route('/process_uploaded_video', methods=['POST'])
def process_uploaded_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    print("video is found")
    file = request.files['video']

    temp_path = os.path.join('static', 'temp_videos', 'temp_video.webm')
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    print("file saved to:", temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        recognizer = HandGestureRecognition(
            model_path='MLP_Keras_model.h5',
            label_encoder_path='label_encoder.pkl'
        )

        pred_sentence = []
        count = 0
        prev_char = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, predicted_char = recognizer.process_frame(frame)
            print("Predicted:", predicted_char)
            if predicted_char:
                if predicted_char == prev_char:
                    count += 1
                else:
                    count = 1
                if count >= 10 and (not pred_sentence or predicted_char != pred_sentence[-1]):
                    pred_sentence.append(predicted_char)
                prev_char = predicted_char

        cap.release()
        os.remove(temp_path)

        sentence = ' '.join(pred_sentence)
        print("Detected Sentence:", sentence)
        return jsonify({'sentence': sentence})

    except Exception as e:
        print(f"Video processing error: {e}")
        return jsonify({'error': 'Processing failed', 'details': str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video = request.files.get('video')
    if video:
        save_path = os.path.join('static', 'uploads', video.filename)
        video.save(save_path)
        return jsonify({"status": "saved", "path": save_path})
    return jsonify({"status": "no file"})

@app.route('/start_camera')

def start_camera():

    print("[INFO] Frontend requested camera access (handled via browser).")
    return jsonify({"status": "camera ready"})

@app.route('/stop_camera')
def stop_camera():
    print("record stop")
    print("[INFO] Frontend closed the camera.")
    return jsonify({"status": "camera stopped"})

# @app.route('/video_feed')
# def video_feed():
#     def generate():
#         global output_frame, lock
#         while camera_active:
#             with lock:
#                 if output_frame is None:
#                     continue
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')

#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_final_sentence')
def get_final_sentence():
    return jsonify({"text": " ".join(final_sentence)})

@app.route('/update_final_sentence', methods=['POST'])
def update_final_sentence():
    global final_sentence
    data = request.get_json()
    text = data.get("text", "")
    final_sentence = text.strip().split()
    return jsonify(success=True)



# Add new route for profile editing
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_info' not in session:
        return redirect(url_for('login'))

    user_info = session['user_info']
    profile_pic_path = user_info.get('profile_picture')
    print(profile_pic_path," path")
    if request.method == 'POST':
        username = request.form['username']
        email = request.form.get('email')
        gender = request.form.get('gender')

        file = request.files.get('profile_picture')
        profile_pic_path = user_info.get('profile_picture')
        print(profile_pic_path," path")
        # Handle new profile picture upload
        if file and file.filename != "" and allowed_file(file.filename):
            filename = secure_filename(f"{user_info['mobile']}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            profile_pic_path = file_path
            profile_pic_path = '/static' +file_path.split('static')[-1].replace('\\', '/') if file_path else None

            # Delete old profile picture if exists
            old_pic = user_info.get('profile_picture')
            if old_pic and os.path.exists(old_pic):
                os.remove(old_pic)

        # Update user in database
        if db_helper.update_user_profile(
            user_info['mobile'],
            username,
            email,
            gender,
            profile_pic_path
        ):
            # Update session data
            session['user_info'] = {
                "mobile": user_info['mobile'],
                "username": username,
                "email": email,
                "gender": gender,
                "profile_picture": profile_pic_path,
            }
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('teacher_dashboard'))
        else:
            flash('Error updating profile', 'error')

    return render_template('profile.html', user_info=user_info)

@app.route('/convert_to_isl', methods=['POST'])
def convert_to_isl():

    data = request.get_json()
    message = data.get("text", "").strip()
    print("convert isl fuction called with ",message)
    if not message:
        return jsonify({"error": "No text provided"}), 400

    # # Create a directory for temporary video files if it doesn't exist
    # temp_video_dir = os.path.join(app.static_folder, 'temp_videos')
    # os.makedirs(temp_video_dir, exist_ok=True)

    # Generate a unique filename for the combined video
    # timestamp = int(time.time())
    # output_filename = f"isl_{timestamp}.mp4"
    # output_path = os.path.join(temp_video_dir, output_filename)

    try:
        # 1. First translate the message to English (if needed)
        # translator = Translator()
        # translated = translator.translate(message, src='auto', dest='en').text
        # print("translatin done : ",translated)
        # 2. Process the text through your ISL conversion pipeline
        compressed = compress_sentence(message.lower())
        handler = TranslationHandler()
        sentences = TextProcessor.split_sentence(compressed)
        print("compression and all done : ",sentences)
        video_paths = []

        # 3. For each word, find the corresponding ISL video
        for sentence in sentences:
            handler.process_sentence(sentence)
            indexed_list = handler.create_indexed_list_with_duplicates()
            print("index list: ",indexed_list)
            conn = DatabaseHandler.connect_to_db()

            for index, word in indexed_list:
                # Look up word in database
                result = DatabaseHandler.lookup_word_with_duration(conn, word)
                print("result: ",result)
                if result:
                    video_paths.append(result[0])  # Append the video path
                else:
                    # Try synonyms if direct word not found
                    synonyms = handler.get_synonyms([word], pos_filter='v')
                    synonym_found = False

                    for synonym in synonyms[word]:
                        syn_result = DatabaseHandler.lookup_synonym_with_duration(conn, synonym)
                        if syn_result:
                            video_paths.append(syn_result[0])
                            synonym_found = True
                            break

                    if not synonym_found:
                        # Fall back to spelling the word
                        alphabet_videos = DatabaseHandler.lookup_alphabet_videos_from_db(conn, word)
                        if alphabet_videos:
                            video_paths.extend([v[0] for v in alphabet_videos])

            conn.close()
        print(video_paths)
        # 4. Return the video paths
        return jsonify({
            "mmmmmmm": video_paths,
            "original_text": message,
            "translated_text": message
        })

    except Exception as e:
        app.logger.error(f"Error converting to ISL: {str(e)}")
        return jsonify({
            "error": "Failed to convert message to ISL",
            "details": str(e)
        }), 500


# Endpoint to handle feedback submission
@app.route('/feedback_message', methods=['POST'])
def feedback_message():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Extract message ID and feedback from the incoming data
        message_id = data.get('message_id')
        feedback = data.get('feedback')

        if not message_id or not feedback:
            return jsonify({'success': False, 'message': 'Missing required fields: message_id or feedback'})


        # Save feedback to the database
        print(f"{message_id}  meassage id  and  {feedback} feedback")
        result = db_helper.StoreFeedback(message_id, feedback)

        return jsonify({'success': result['success'], 'message': result['message']})

    except Exception as e:
        # Handle any exceptions and return an error response
        return jsonify({'success': False, 'message': str(e)})
# # Add this cleanup function to remove old temporary videos
# def cleanup_temp_videos():
#     temp_video_dir = os.path.join(app.static_folder, 'temp_videos')
#     if os.path


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

    app.run(debug=True)
