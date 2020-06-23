# Angelina braille reader

Draft version. Production version will be at https://github.com/IlyaOvodov/AngelinaReader (under construction)


# requirements

 ubuntu, windows with GPU  

 CUDA  
 Python 3.6  
 PyTorch 1.4  
 torchvision  
 ignite  
 numpy  
 PIL  
 albumentations  
 cv2  
 https://github.com/IlyaOvodov/pytorch-retinanet  
 https://github.com/IlyaOvodov/OvoTools  
 https://github.com/IlyaOvodov/labelme  (for annotation)

# installation

`git clone https://github.com/IlyaOvodov/BrailleV0.git  
 edit BrailleV0/local_config.py to set data_path and global_3rd_party pointing to current dir  
 git clone https://github.com/IlyaOvodov/OvoTools.git  
 cd OvoTools  
 python setup.py develop  
 cd ..  
 git clone https://github.com/IlyaOvodov/pytorch-retinanet.git pytorch_retinanet  
 download model https://yadi.sk/d/GW0qEmA5rL0m0A into ./NN_saved/retina_chars_eced60/models/clr.008
`

# usage
`cd BrailleV0/web-app
 python angelina_reader_app.py
 access it by 127.0.0.1:5000
`

or edit `if __name__=="__main__"` section BrailleV0/NN/RetinaNet/infer_retinanet.py. Set img_filename_mask and results_dir to proper values and run it.