# DetetronCheck
 Model to segment and classify check boxes

OS: Linux distro (ubuntu or mint will do)

# installation

* If you dont have linux install [Mint](https://www.linuxmint.com/download.php) in a virtual box (give it atleast 6 gig ram)
* Install [anaconda](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh) in it. once its downloaded run:
 
```bash
>> bash Anaconda3-2020.07-Linux-x86_64.sh
```
* To update anaconda:
```bash
>> conda update -n env conda
```
* install the following dependencies by running these commands in your terminal:
```bash
>> pip install pyyaml==5.1 pycocotools>=2.0.1
>> pip install opencv-python
>> pip install pdf2image
>> conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
>> pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

* When they are done test that you have detectron2 installed:
```bash
>> python
>> import detectron2
>> exit()
```
if it executes without error you are good to go.

# The app
* Download and unzip the delivery.
* cd to DetectronCheck/scr then in the terminal run:
```bash
>> python app.py
```

If everything is installed and matches versions you should be able to open the app in your browser localhost:5000

Once its open,  you will choose a pdf file and click upload. then it wil give a prediction and a csv file will be downloaded into the _DetectronCheck/scr/static/csv_results/_  folder.

* Everything else from here is building around the model and making it scale with good front and backend.
-----


