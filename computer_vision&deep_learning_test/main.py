import os
import cv2
import glob
import func_q1


def face_detect(img, face_cascade):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_pos = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    return faces_pos


def eye_detect(img, eye_cascade):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes_pos = eye_cascade.detectMultiScale(gray_img, 1.3, 10)
    return eyes_pos


def apply_logo(roi_face, logo, ex, ey, ew, eh):
    logo = cv2.resize(logo, (ew, eh), interpolation=cv2.INTER_AREA)
    roi_eye = roi_face[ey:ey+eh, ex:ex+ew]
    for row in range(ew):
        for col in range(eh):
            if logo[row, col].all():
                roi_eye[row, col] = logo[row, col]
    return roi_face


def main():
    model_path = r"./test-bundle/res-cv-test\model-haar"
    sample_path = r"./test-bundle/res-cv-test\samples\blackpink"
    output_path = r"./output"
    face_cascade = cv2.CascadeClassifier(os.path.join(model_path,
                                                      "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(os.path.join(model_path,
                                                     "haarcascade_eye.xml"))
    bp_path = glob.glob(os.path.join(sample_path, "*.jpg"))
    logo_path = glob.glob(os.path.join(sample_path, "*.png"))[0]
    titles = [os.path.basename(title)[:-4] for title in bp_path]
    original_bp_img = [cv2.imread(path) for path in bp_path]
    bp_img = original_bp_img.copy()
    logo_img = cv2.imread(logo_path)
    func_q1.plot_gallery(original_bp_img, titles)
    for idx, img in enumerate(bp_img):
        face_pos = face_detect(img, face_cascade)
        (fx, fy, fw, fh) = face_pos[0]
        roi_face = img[fy:fy+fh, fx:fx+fw]
        # cv2.rectangle(img,
        #               (fx, fy),
        #               (fx+fw, fy+fh),
        #               (255, 0, 0), 2)

        eyes_pos = eye_detect(roi_face, eye_cascade)
        for (ex, ey, ew, eh) in eyes_pos:
            # cv2.rectangle(roi_face,
            #               (ex, ey),
            #               (ex+ew, ey+eh),
            #               (0, 255, 0),
            #               2)
            roi_face = apply_logo(roi_face, logo_img, ex, ey, ew, eh)
        # cv2.imshow(titles[idx], img)
        img_path = os.path.join(output_path, titles[idx]+".jpg")
        cv2.imwrite(img_path, img)
    func_q1.plot_gallery(bp_img, titles)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
