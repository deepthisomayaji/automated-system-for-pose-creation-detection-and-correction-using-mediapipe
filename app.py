#conda activate main
#streamlit run app.py

import os
import cv2
import yaml
import numpy as np
from PIL import Image
import mediapipe as mp
import streamlit as st
from keras.models import Model
from yaml.loader import SafeLoader
from keras.layers import Input, Dense
from streamlit_option_menu import option_menu
from streamlit_authenticator import Authenticate
from tensorflow.keras.utils import to_categorical

os.chdir('D:/mtechsem4/Project/infidata/yoga_pose_delivery')

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login(fields='main')

if authentication_status:
    if username == 'admin':
        st.title('ADMIN PAGE')
        with st.sidebar:
            admin_page_option = option_menu('NAVIGATION',
                                            options=['HOME', 'ADD POSE', 'TRAIN MODEL'],
                                            icons = ['house', 'plus-lg', 'clipboard-check'])

        
        current_yoga_poses = os.listdir('data')
        

        if admin_page_option == 'HOME':
            st.warning('This project leverages deep learning and computer vision technologies to develop a yoga pose detection tool. By utilizing OpenCV for image processing and deep learning models for pose evaluation, the project aims to create an accessible and effective solution for yoga practitioners. Additionally, MediaPipe is employed for pose landmark detection, enhancing the accuracy of pose recognition and feedback. The project is built on a Streamlit framework, offering a user-friendly interface and a role-based login system.')
            st.error('This system distinguishes between administrators and end users, providing tailored functionalities for each. Administrators can add new yoga poses and provide feedback by performing the poses in front of the camera. MediaPipe is utilized to capture and process the pose landmarks, which are then used to train Artificial Neural Network (ANN) models. Each yoga pose has its own separately trained ANN model to ensure precise evaluation.')
        
        if admin_page_option == 'ADD POSE':
            st.selectbox('Current Yoga Poses List : ',
                                                options=current_yoga_poses)
            st.markdown('---')

            new_yoga_pose_input = st.text_input("Enter Yoga Pose Name : ")

            st.markdown('---')

            if new_yoga_pose_input:
                if new_yoga_pose_input.lower() in current_yoga_poses:
                    # st.subheader(f"these are the feedbacks in {new_yoga_pose_input}")

                    feedback_list = os.listdir(rf'data/{new_yoga_pose_input.lower()}')
                    feedback_list_without_extension = [file.split('.')[0] for file in feedback_list]
                    st.selectbox('FEEDBACK LIST',options=feedback_list_without_extension)
                    st.error(f'{new_yoga_pose_input} already found in database. You can add feedbacks to it.')

                    st.markdown('---')
                    new_feedback = st.text_input("Enter new feedback to add")
                    add_feedback_button = st.button('ADD FEEDBACK')

                    if add_feedback_button:
                        if new_feedback:
                            FRAME_WINDOW = st.image([])

                            if new_feedback not in feedback_list_without_extension:
                                # Function to check if the pose is within frame
                                def check_pose(lst):
                                    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
                                        return True
                                    return False


                                # Open the video capture device (webcam)
                                cap = cv2.VideoCapture(0)

                                # Get the name of the Asana or Feedback from the user
                                name = new_feedback

                                # Initialize the pose detection module
                                holistic = mp.solutions.pose
                                holis = holistic.Pose()
                                drawing = mp.solutions.drawing_utils

                                # Initialize an empty list to store the pose landmarks
                                X = []

                                # Initialize the variable to keep track of the number of data points collected
                                data_size = 0

                                # Main loop to capture video frames and detect poses
                                while True:
                                    pose_landmarks = []

                                    # Capture a frame from the video feed
                                    _, frame = cap.read()

                                    # Flip the frame horizontally
                                    frame = cv2.flip(frame, 1)

                                    # Process the frame to detect the pose landmarks
                                    results = holis.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                                    # If pose landmarks are detected and the pose is within frame, add the pose landmarks to the list
                                    if results.pose_landmarks and check_pose(results.pose_landmarks.landmark):
                                        for i in results.pose_landmarks.landmark:
                                            pose_landmarks.append(i.x - results.pose_landmarks.landmark[0].x)
                                            pose_landmarks.append(i.y - results.pose_landmarks.landmark[0].y)

                                        # Add the pose landmarks to the list of X
                                        X.append(pose_landmarks)

                                        # Increment the data size
                                        data_size = data_size + 1

                                    # If the pose is not within frame, display a warning message
                                    else:
                                        cv2.putText(frame, "WARNING : Full Body Not Visible!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                    # Draw the pose landmarks on the frame
                                    drawing.draw_landmarks(frame, results.pose_landmarks, holistic.POSE_CONNECTIONS)

                                    # Display the number of data points collected on the frame
                                    cv2.putText(frame, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                                    # Convert the frame from BGR to RGB
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    FRAME_WINDOW.image(rgb_frame)

                                    # Exit the loop if the 'Esc' key is pressed or the number of data points collected exceeds 180
                                    if cv2.waitKey(1) == 27 or data_size > 100:
                                        cv2.destroyAllWindows()
                                        cap.release()
                                        break

                                # Save the collected pose landmarks to a numpy file
                                np.save(f"data/{new_yoga_pose_input}/{name}.npy", np.array(X))


                            else:
                                st.error('feedback already exists')
                        else:
                            st.error('no text label found for new feedback/feedback already present')
                    else:
                        st.error('Click submit.')

                else:
                    st.error(f'{new_yoga_pose_input} not found in Database, you can add it now.')

                    add_yoga_pose_button = st.button("ADD YOGA POSE")

                    if add_yoga_pose_button:
                        st.success(f"added **{new_yoga_pose_input}** to database...")
                        os.mkdir(f'data/{new_yoga_pose_input}')
                    else:
                        st.error('CLICK ADD POSE BUTTON!')
                

        
        if admin_page_option == 'TRAIN MODEL':
            st.header("TRAIN A ANN ON DATASET")
            yoga_pose_name = st.selectbox('Enter Name of the Yoga Pose to train : ',
                                          options=current_yoga_poses)

            train_model = st.button('TRAIN MODEL')
            if train_model:
                if yoga_pose_name.lower() in current_yoga_poses:

                    def train_ann_model(yoga_pose_name):
                        is_initialized = False
                        size_data = -1

                        label_names = []
                        label_dict = {}
                        class_count = 0

                        base_directory = f'D:/mtechsem4/Project/infidata/yoga_pose_delivery/'
                        data_directory = f'D:/mtechsem4/Project/infidata/yoga_pose_delivery/data/'
                        current_directory = os.path.join(data_directory, yoga_pose_name)
                        os.chdir(current_directory)
                        # os.chdir(current_directory)
                        if os.listdir(current_directory):
                            try:
                                # Iterate over files in the directory
                                for file_name in os.listdir(current_directory):
                                    # Check if the file is a numpy file and not the label file
                                    if file_name.split(".")[-1] == "npy" and not (file_name.split(".")[0] == "labels"):

                                        # Load the data from the numpy file
                                        if not (is_initialized):
                                            is_initialized = True

                                            # st.info(os.listdir(current_directory))
                                            # st.info(np.load(file_path))

                                            X_data = np.load(file_name)
                                            size_data = X_data.shape[0]
                                            y_data = np.array([file_name.split('.')[0]] * size_data).reshape(-1, 1)
                                        else:
                                            X_data = np.concatenate((X_data, np.load(file_name)))
                                            y_data = np.concatenate((y_data, np.array([file_name.split('.')[0]] * size_data).reshape(-1, 1)))

                                        # Store the class name in the list
                                        label_names.append(file_name.split('.')[0])
                                        # Store the class name and index in the dictionary
                                        label_dict[file_name.split('.')[0]] = class_count
                                        class_count += 1

                                # Convert the class labels to categorical format
                                for i in range(y_data.shape[0]):
                                    y_data[i, 0] = label_dict[y_data[i, 0]]
                                y_data = np.array(y_data, dtype="int32")

                                y_data = to_categorical(y_data)

                                X_new_data = X_data.copy()
                                y_new_data = y_data.copy()
                                counter_data = 0

                                # Shuffle the data
                                count_arr = np.arange(X_data.shape[0])
                                np.random.shuffle(count_arr)

                                # Rearrange the data based on the shuffled indices
                                for i in count_arr:
                                    X_new_data[counter_data] = X_data[i]
                                    y_new_data[counter_data] = y_data[i]
                                    counter_data += 1

                                #creating an ANN model
                                st.info('training the model...')

                                print(X_data.shape[1])
                                input_layer = Input(shape=(X_data.shape[1],))
                                model_layer = Dense(128, activation="tanh")(input_layer)
                                model_layer = Dense(64, activation="tanh")(model_layer)
                                output_layer = Dense(y_data.shape[1], activation="softmax")(model_layer)
                                model = Model(inputs=input_layer, outputs=output_layer)

                                #compiling the model
                                model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

                                #flask the model
                                model.fit(X_new_data, y_new_data, epochs=80)

                                model.save(f"{yoga_pose_name}_model.h5")#saving the trained model
                                np.save(f"{yoga_pose_name}_labels.npy", np.array(label_names))

                                st.success('MODEL TRAINING COMPLETE...')
                                os.chdir(base_directory)
                            finally:
                                os.chdir(base_directory)
                        else:
                            st.error('No feedback found')
                            os.chdir(base_directory)
                        os.chdir(base_directory)

                    train_ann_model(yoga_pose_name)

    
    else:
        st.title('YOGA POSE DETECTION TOOL')
        with st.sidebar:
            end_user_page_option = option_menu('NAVIGATION',
                                            options=['ABOUT','INFORMATION', 'YOGA GURU'],
                                            icons=['house','info-square', 'person-check'])
        
        if end_user_page_option == 'ABOUT':
            st.warning('This project leverages deep learning and computer vision technologies to develop a yoga pose detection tool. By utilizing OpenCV for image processing and deep learning models for pose evaluation, the project aims to create an accessible and effective solution for yoga practitioners. Additionally, MediaPipe is employed for pose landmark detection, enhancing the accuracy of pose recognition and feedback. The project is built on a Streamlit framework, offering a user-friendly interface and a role-based login system.')
            st.error('This system distinguishes between administrators and end users, providing tailored functionalities for each. Administrators can add new yoga poses and provide feedback by performing the poses in front of the camera. MediaPipe is utilized to capture and process the pose landmarks, which are then used to train Artificial Neural Network (ANN) models. Each yoga pose has its own separately trained ANN model to ensure precise evaluation.')
        
        
        if end_user_page_option == 'INFORMATION':
            st.header('TREE POSE')
            tree_pose_image = Image.open('images/tree_pose.png')
            st.image(tree_pose_image)
            st.warning('Tree Pose is a strengthening posture that can help build confidence. This pose can improve your posture and counteract the effects of prolonged sitting. On your standing leg, this pose strengthens your thigh, buttock (glute), and ankle. On your lifted leg, this pose gently stretches your entire thigh and buttocks.')
            
            st.subheader('HOW TO PERFORM TREE POSE')
            st.info('1. Stand in Tadasana. Spread your toes, press your feet into the mat and firm your leg muscles. Raise your front hip points toward your lower ribs to gently lift in your lower belly.')
            st.error('2. Inhale deeply, lifting your chest, and exhale as you draw your shoulder blades down your back. Look straight ahead at a steady gazing spot.')
            st.warning('3. Place your hands on your hips and raise your right foot high onto your left thigh or shin. Avoid making contact with the knee.')
            st.info('4. Press your right foot and left leg into each other.')
            st.error('5. Check that your pelvis is level and squared to the front.')
            st.warning('6. When you feel steady, place your hands into Anjali Mudra at the heart or stretch your arms overhead like branches reaching into the sun.')
            st.success('7. Hold for several breaths, then step back into Mountain Pose and repeat on the other side.')

        if end_user_page_option == 'YOGA GURU':
            current_yoga_poses = os.listdir('data')
            yoga_guru_pose_input = st.selectbox('Current Yoga Poses List : ',
                                                options=current_yoga_poses)
            
            FRAME_WINDOW = st.image([])
            from keras.models import load_model

            yoga_guru_button = st.button('CHECK POSE')
            if yoga_guru_button:
                def inFrame(lst):
                    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
                        return True
                    return False

                base_directory = f'D:/mtechsem4/Project/infidata/yoga_pose_delivery'
                destination_directory = f'{base_directory}/data/{yoga_guru_pose_input}'


                
                try:
                    os.chdir(destination_directory)
                    model = load_model(f"{destination_directory}/{yoga_guru_pose_input}_model.h5")
                    label = np.load(f"{destination_directory}/{yoga_guru_pose_input}_labels.npy")
        
                    holistic = mp.solutions.pose
                    holis = holistic.Pose()
                    drawing = mp.solutions.drawing_utils

                    cap = cv2.VideoCapture(0)

                    try:

                        while True:
                            lst = []

                            _, frm = cap.read()

                            # window = np.zeros((940,940,3), dtype="uint8")

                            frm = cv2.flip(frm, 1)

                            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

                            frm = cv2.blur(frm, (4, 4))
                            if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
                                for i in res.pose_landmarks.landmark:
                                    lst.append(i.x - res.pose_landmarks.landmark[0].x)
                                    lst.append(i.y - res.pose_landmarks.landmark[0].y)

                                lst = np.array(lst).reshape(1, -1)

                                p = model.predict(lst)
                                pred = label[np.argmax(p)]

                                if p[0][np.argmax(p)] > 0.75:
                                    # cv2.putText(window, pred , (180,180),cv2.FONT_ITALIC, 1.3, (0,255,0),2)
                                    cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)

                                else:
                                    # cv2.putText(window, "Asana is either wrong not trained" , (100,180),cv2.FONT_ITALIC, 1.8, (0,0,255),3)
                                    cv2.putText(frm, "Activity is either wrong or not trained", (100, 90), cv2.FONT_ITALIC, 0.8, (0, 0, 255), 3)

                            else:
                                # cv2.putText(frm, "Make Sure Full body visible", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)
                                cv2.putText(frm, "Make Sure Full body is visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                            drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                                connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                                                landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

                                    # Convert the frame from BGR to RGB
                            rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(rgb_frame)
                    finally:
                        os.chdir(base_directory)
                finally:
                    os.chdir(base_directory)
                
                    


    authenticator.logout('Log out','sidebar')

elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username and password')

