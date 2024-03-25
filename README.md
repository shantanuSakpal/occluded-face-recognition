# face recognition

This face recognition implementation is capable of recognizing faces with a certain level of occlusion, this includes faces wearing masks.
You can also add new users manually by adding a photo in the images folder.

# How to run:

<pre><code>python main.py --input webcam</code></pre>

![Results](https://github.com/juan-csv/face_recognition_occlusion/blob/master/results/result.gif)

if you don't want to run it with the webcam use

<pre><code>python main.py --input image --path_im test_image.jpeg</code></pre>

# Add new faces to the database (facial recognition)

You can add new users to the faces database simply by adding the person's photo in format .jpg in the **images** folder, for the registry to work correctly, only the person of interest should appear in the photo.

# References

- **Face recognition:** https://github.com/ageitgey/face_recognition

# How to run the model

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:

- For Windows: `venv\Scripts\activate`
- For macOS/Linux: `source venv/bin/activate`

3. Install the dependencies: `pip install -r requirements.txt`
4. Start the backend server: `python app.py`
