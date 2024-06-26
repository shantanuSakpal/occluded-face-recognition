# face recognition

This face recognition implementation is capable of recognizing faces with a certain level of occlusion, this includes faces wearing masks.
You can also add new users manually by adding a photo in the images folder.

# Add new faces to the database (facial recognition)

You can add new users to the faces database simply by adding the person's photo in format .jpg in the **images** folder, for the registry to work correctly, only the person of interest should appear in the photo.

# References

- **Face recognition:** https://github.com/ageitgey/face_recognition

# How to set up the environment:

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:

- For Windows: `venv\Scripts\activate`
- For macOS/Linux: `source venv/bin/activate`

3. Install the dependencies: `pip install -r requirements.txt`

# How to run:

To run with webcam, use:

<pre><code>python main.py --input webcam</code></pre>

To run with a single image use:

<pre><code>python main.py --input image --path_im test_image.jpg</code></pre>

To run with a folder containing test images use:

<pre><code>python main.py --input image --folder_path ./test_images</code></pre>

### To Do

1. how does og face_recognition work.(algorithm) - uday
2. how to show that it doesnt work for occluded face. -uday
3. how does ours work.(algorithm) -shantanu
4. how to show that ours work for occluded face. -shantanu
5. future use case:
   a. synthecial put a mask on wanted people faces. and give it to our model.
   b. remove occlusion from the face. if unknow person using silp / gan - divyesh

So basically this whole project is to detect masked/occluded faces, once we detect masked/occluded faces (mark them with bounding boxes), we can find the features of the face.
Once we have the features, we can:

1. Remove the mask/occlusion from the face.
2. Recognize face with higher accuracy.

We will be focusing on second point.
This is the main idea of the project.

now, the og face_recognition library is not able to detect occluded faces, so we will be using our own model to detect occluded faces.
and then we will make bounding boxes around the occluded faces and then we will use the og face_recognition library to detect the faces, by providing it our bounding boxes.

Precision: The fraction of detected faces that are correctly identified as faces.
Recall: The fraction of actual faces that are correctly detected.
F1-score: The harmonic mean of precision and recall, providing a balanced measure of performance.
Intersection over Union (IoU): A measure of how well the predicted bounding boxes overlap with the ground truth bounding boxes.
