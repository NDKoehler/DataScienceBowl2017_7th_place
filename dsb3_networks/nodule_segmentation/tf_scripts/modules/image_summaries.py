import numpy as np
import cv2
import os, sys


def segmentation_image_summary(kwargs, **inputs):
  
    num_img_logs = 10
       
    img_logs = []
    for i in range(num_img_logs):
        batch_id = np.random.randint(0, kwargs['batch_size'])

        #image log
        images = inputs['images'][batch_id].copy()
     
        # central channel
        central_channel_image = int(kwargs['image_shape'][2]/2) # taking central channel (=layer)
        images = images[:,:,central_channel_image]

        images = images + 0.25
        images *= 255
        images = images.astype(np.uint8)

        images = cv2.cvtColor( images, cv2.COLOR_GRAY2BGR )
        #cv2.imwrite('images.jpg', images)
        img_logs.append((images, inputs['mode'] + '_' + str(num_img_logs*i + 0)))

        #label log
        labels = inputs['labels'][batch_id].copy()

        labels = labels * 255
        size = (kwargs['label_shape'][0], kwargs['label_shape'][1], 3)
        labels = labels.reshape(size)
        labels = labels.astype(np.uint8)
        #cv2.imwrite('lab.jpg', labels)

        # labels = cv2.cvtColor( labels, cv2.COLOR_GRAY2BGR )

        img_logs.append((labels, inputs['mode'] + '_' + str(num_img_logs*i +1)  ))

        #prediction log
        mask = inputs['probs'][batch_id].copy()
        mask = mask * 255
        size = (kwargs['label_shape'][0], kwargs['label_shape'][1], 1)
        mask = mask.reshape(size)
        mask = mask.astype(np.uint8)   
        mask = cv2.cvtColor( mask, cv2.COLOR_GRAY2BGR )
        img_logs.append((mask, inputs['mode'] + '_' + str(num_img_logs*i + 2)   ))

        #error log
        pred_mask = inputs['probs'][batch_id].copy()
        size = (kwargs['label_shape'][0], kwargs['label_shape'][1], 1)
        pred_mask = pred_mask.reshape(size)
        label = inputs['labels'][batch_id].copy()
        label = np.reshape(label, (kwargs['label_shape'][0], kwargs['label_shape'][1], 3))
        error_map = 1.0 - np.abs(pred_mask - label[:,:,:1])
        error_map = error_map * 255
        error_map = error_map.astype(np.uint8)
        error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
        img_logs.append((error_map, inputs['mode'] + '_' + str(num_img_logs*i +3)   ))


    return img_logs
