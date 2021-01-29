import cv2
import sys

Kaskad = sys.argv[1]
FaceKaskad = cv2.CascadeClassifier(Kaskad)
######Создание каскада лица
VideoZahvat = cv2.VideoCapture(0)
####вебка как источник
while True:
    ###########покадровое чтение отснятного маткериала
    ret, Kadr = VideoZahvat.read()

    gray = cv2.cvtColor(Kadr, cv2.COLOR_BGR2GRAY)

    faces = FaceKaskad.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    ###### Рисуем прямоугольник
    for (x, y, w, h) in faces:
        cv2.rectangle(Kadr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    ####### Демонстрация остаточного кадра
    cv2.imshow('Video', Kadr)
    #####ищем личеко и рвём при нажание клавиши
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

###########Когда готово просто выпускаем из фокуса обьект
VideoZahvat.release()
cv2.destroyAllWindows()

#использовать через:
#############python filename.py haarcascade_frontalface_default.xml