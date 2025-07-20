import os
import cv2 as cv
data_directory = "data/raw"

def data_loader_and_process():
    images = []
    images_files = [i for i in os.listdir(data_directory)]
    processed_folder = "data/processed"
    counts = {}
    for image in images_files:
        img_directory = data_directory + image
        captcha = image[0:4]
        img = cv.imread(img_directory, cv.IMREAD_GRAYSCALE)
        img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_REPLICATE)
        inverted = cv.bitwise_not(img)
        _, threshold_image = cv.threshold(inverted, 0, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(threshold_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv.boundingRect(i) for i in contours]
        contours_sorted = sorted(bounding_boxes, key = lambda x: x[0])

        letter_image_regions = []

        for cnt in contours_sorted:
            x, y, w, h = cnt
            # cv.rectangle(img_coloured, (x,y), (x+w, y+h), (0, 255, 0), 1)
            if w/h > 1.25:
                half_width = int(w/2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))
        
        if len(letter_image_regions) != 4:
            continue
        
        for letter_contour_box, letter in zip(letter_image_regions, captcha):
            x, y, w, h = letter_contour_box
            roi = img[y:y+h, x:x+w]

            output_path = os.path.join(processed_folder, letter)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            count = counts.get(letter, 1)
            image_path = os.path.join(output_path, f'{str(count).zfill(6)}.png')
            
            cv.imwrite(image_path, roi)

            counts[letter] = count + 1