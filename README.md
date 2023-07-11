# Companionship Robot for People with Anxiety

This project implements facial emotion analysis using a Raspberry Pi, OpenCV, and a trained deep learning model. It captures video from a webcam, detects faces using Haar cascades, and classifies the detected faces into different emotions using a pre-trained model. Depending on the predicted emotion, the Raspberry Pi's motor control module is used to perform specific actions.

## Dependencies

- [OpenCV](https://raspberrypi-guide.github.io/programming/install-opencv)
- [Keras](https://www.teknotut.com/install-tensorflow-and-keras-on-the-raspberry-pi/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.samwestby.com/tutorials/rpi-tensorflow.html)
- [RPi.GPIO](https://pypi.org/project/RPi.GPIO/)

## Steps to run the code
1. Git clone this project into your local machine
2. Upload the code into your rasberry pi
3. Run these commands and collect the info

```python
cat /etc/os-release
sudo apt update
sudo apt upgrade
```
4. Find your .sh script -- python3 -V (take a note of this) uname -m (take a note of this)
5. Check if there is a shell file for your Python/Architecture combo here: [Link](https://github.com/PINTO0309/Tensorflow-bin/tree/main/previous_versions)
- If python3 -V = "3.9.*" (* means any number) AND uname -m = "aarch64"
6. Download your respective package into the raspberry pi
7. Install system packages

```python
sudo apt-get install --yes libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libgdbm-dev lzma lzma-dev tcl-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev wget make openssl
```
8. Update pyenv
```python
pyenv update
```
9.  Install python versions
```python
pyenv install --list
pyenv install ~~Your python version~~
```
10. Repeat the necessary process for the respective packages 
11. Locate your code on the raspberry pi terminal and run the code

## Reference Links
- [Link_1](https://www.youtube.com/watch?v=ufzptG4rMHk)
- [Link_2](https://www.youtube.com/watch?v=Cj7NhuLkvdQ)
- [Link_3](https://www.youtube.com/watch?v=vekblEk6UPc&t=870s)
