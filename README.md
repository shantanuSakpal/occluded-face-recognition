# face recognition

This face recognition implementation is capable of recognizing faces with a certain level of occlusion, this includes faces wearing masks.
You can also add new users manually by adding a photo in the images folder.

# How to run:

<pre><code>python main.py --input webcam</code></pre>

//add gif here

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

### To Do

1. how does og face_recognition work.(algorithm) - uday
2. how to show that it doesnt work for occluded face. -uday
3. how does ours work.(algorithm) -shantanu
4. how to show that ours work for occluded face. -shantanu
5. future use case:
   a. synthecial put a mask on wanted people faces. and give it to our model.
   b. remove occlusion from the face. if unknow person using silp / gan - divyesh
