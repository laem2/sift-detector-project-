import cv2
import matplotlib.pyplot as plt
import os
import tkinter as tk

dataset = r"dataset/ME"
dataset_images =[]

for image_name in os.listdir(dataset):
    if image_name.endswith('.jpg'):
        image_path = os.path.join(dataset, image_name)
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)


        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        dataset_images.append((image, keypoints,descriptors))


query_path = r"testset\me_cropped\noisier.jpg"
query_image = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
# #######################################################################################
bf = cv2.BFMatcher(cv2.NORM_L2)
current_best_matches = 0
for image, keypoints, descriptors in dataset_images:
    matches = bf.knnMatch(query_descriptors, descriptors, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
        
    if len(good_matches) > current_best_matches:
        current_best_matches = len(good_matches)
        best_match = (image, keypoints, sorted(good_matches, key=lambda x: x.distance))

top_matches = best_match[2][:50]

image_match = cv2.drawMatchesKnn(query_image,query_keypoints,best_match[0],best_match[1],
                                 [top_matches],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure()
plt.imshow(image_match)
plt.show()

