import numpy as np
import cv2


def classification_image_summary(kwargs, **inputs):
  
    num_img_logs = kwargs['batch_size']
       
    img_logs = []
    for i in range(num_img_logs):
        
        batch_id = i#np.random.randint(0, kwargs['batch_size'])
        candidate_id = np.random.randint(0, inputs['images'].shape[1])

        prob = inputs['probs'][batch_id][0].copy()
        lab = inputs['labels'][batch_id][0].copy()

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

        images = cv2.resize(images, (256,256))

        font = cv2.FONT_HERSHEY_SIMPLEX

        color = (0,255,0) if lab == 0 else (255,0,0)
        cv2.putText(images,str(np.round(prob, 4)), (10,30), font, 1,color,2)


        img_logs.append((images, inputs['mode'] + '_' + str(num_img_logs*i + 0) +'_p_'+ str(prob) ))

    return img_logs
