import numpy as np
import cv2



def segmentation_image_summary(kwargs, **inputs):
  
    num_img_logs = 10
       
    img_logs = []
    for i in range(num_img_logs):
        batch_id = np.random.randint(0, kwargs['batch_size'])


        #image log
        images = inputs['images'][batch_id].copy()
        if (kwargs['image_shape'][2] != 1 and kwargs['image_shape'][2] != 3):
            rand_channel_image = np.random.randint(0, kwargs['image_shape'][2])
            images = images[:,:,rand_channel_image]
            images = np.expand_dims(images,2)

        images = images * 128.0
        images += 128.0
        images = images.astype(np.uint8)

        if (images.shape[2] == 1):
            images = cv2.cvtColor( images, cv2.COLOR_GRAY2BGR )

        img_logs.append((images, inputs['mode'] + '_' + str(num_img_logs*i + 0)))

        #label log
        if (kwargs['label_shape'][2] != 1):
            rand_channel_label = np.random.randint(0, kwargs['label_shape'][2])
        else:
            rand_channel_label = 0

        labels = inputs['labels'][batch_id].copy()
        labels = labels * 255
        size = (kwargs['label_shape'][0], kwargs['label_shape'][1], kwargs['label_shape'][2])
        labels = labels.reshape(size)
        labels = labels.astype(np.uint8)

        if (kwargs['label_shape'][2] != 1):
            labels = labels[:,:, rand_channel_label].copy()
        labels = cv2.cvtColor( labels, cv2.COLOR_GRAY2BGR )

        img_logs.append((labels, inputs['mode'] + '_' + str(num_img_logs*i +1)  ))

        #prediction log
        mask = inputs['probs'][batch_id].copy()
        mask = mask * 255
        size = (kwargs['label_shape'][0], kwargs['label_shape'][1], kwargs['label_shape'][2])
        mask = mask.reshape(size)
        mask = mask.astype(np.uint8)   
        if (kwargs['label_shape'][2] != 1):
            mask = mask[:,:, rand_channel_label].copy()
        mask = cv2.cvtColor( mask, cv2.COLOR_GRAY2BGR )
        img_logs.append((mask, inputs['mode'] + '_' + str(num_img_logs*i +2)   ))

        #error log
        pred_mask = inputs['probs'][batch_id].copy()
        size = (kwargs['label_shape'][0], kwargs['label_shape'][1], kwargs['label_shape'][2])
        pred_mask = pred_mask.reshape(size)
        label = inputs['labels'][batch_id].copy()        
        error_map = 1.0 - np.abs(pred_mask - label)        
        error_map = error_map * 255
        error_map = error_map.astype(np.uint8)
        if (kwargs['label_shape'][2] != 1):
            error_map = error_map[:,:, rand_channel_label].copy()
        error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
        img_logs.append((error_map, inputs['mode'] + '_' + str(num_img_logs*i +3)   ))


    return img_logs


def classification_image_summary(kwargs, **inputs):
  


    num_img_logs = 40
       
    img_logs = []
    for i in range(num_img_logs):
        batch_id = np.random.randint(0, kwargs['batch_size'])
        candidate_id = 0#np.random.randint(0, inputs['images'].shape[1])

        prop = inputs['predictions'][batch_id]

        #print(inputs['images'].shape, kwargs['batch_size'])
        #image log
        if len(inputs['images'].shape) == 6:
            images = inputs['images'][batch_id][candidate_id].copy()
        if len(inputs['images'].shape) == 4:
            images = inputs['images'][batch_id].copy()


        #print(images.shape)
        if len(images.shape) == 4:
            images = np.expand_dims(images[images.shape[0]//2,:,:,0],2)
        
        if len(images.shape) == 3:
            images = np.expand_dims(images[:,:,images.shape[2]//2],2)

        images = (images+0.25) * 255.0
        images = images.astype(np.uint8)
        #print("sum of image", np.sum(images))
        #if np.sum(images) == 0:
            #np.save('/media/niklas/Data_2/dsb3/test/zeros.npy', inputs['images'][batch_id])
            #sys.exit()
        if (images.shape[2] == 1):
            images = cv2.cvtColor( images, cv2.COLOR_GRAY2BGR )

        img_logs.append((images, inputs['mode'] + '_' + str(num_img_logs*i + 0) +'_p_'+ str(prop) ))

    return img_logs
