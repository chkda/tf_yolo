import numpy as np 
import tensorflow as tf

class Yolov1():

    def __init__(self,num_classes,anchor_box,cell_size,batch_size,image_size,box_per_cell):

        self.num_classes = num_classes
        self.box_per_cell = box_per_cell
        self.anchor_box = anchor_box
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.image_size = image_size

        self.class_scale = 1.0
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.coordinate_scale = 1.0

        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)]*self.cell_size*self.box_per_cell),[self.box_per_cell,self.cell_size,self.cell_size]),(1,2,0))

        self.offset = tf.reshape(tf.constant(self.offset,dtype='float32'),[1,self.cell_size,self.cell_size,self.box_per_cell])

        self.offset = tf.tile(self.offset,(self.batch_size,1,1,1))


    def yolo_conv(self,inp,filters,ksize,name,batch_norm=True):

        x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(ksize,ksize),strides=(1,1),padding='SAME',name=name)(inp)
        
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyRelu(alpha=0.1)(x)

        return x

    def yolo_mpool(self,inp,name):

        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME',name=name)(inp)

        return x

    def comp_graph(self,inp):

        model = self.yolo_conv(inp,filters=32,ksize=3,name='conv_1')
        model = self.yolo_mpool(model,name='mpool_2')

        model = self.yolo_conv(model,filters=64,ksize=3,name='conv_3')
        model = self.yolo_mpool(model,name='mpool_4')

        model = self.yolo_conv(model,filters=128,ksize=3,name='conv_5')
        model = self.yolo_conv(model,filters=64,ksize=1,name='conv_6')
        model = self.yolo_conv(model,filters=128,ksize=3,name='conv_7')
        model = self.yolo_mpool(model,name='mpool_8')

        model = self.yolo_conv(model,filters=256,ksize=3,name='conv_9')
        model = self.yolo_conv(model,filters=128,ksize=1,name='conv_10')
        model = self.yolo_conv(model,filters=256,ksize=3,name='conv_11')
        model = self.yolo_mpool(model,name='mpool_12')

        model = self.yolo_conv(model,filters=512,ksize=3,name='conv_13')
        model = self.yolo_conv(model,filters=256,ksize=1,name='conv_14')
        model = self.yolo_conv(model,filters=512,ksize=3,name='conv_15')
        model = self.yolo_conv(model,filters=256,ksize=1,name='convv_16')
        model = self.yolo_conv(model,filters=512,ksize=3,name='conv_17')
        model = self.yolo_mpool(model,name='mpool_18')

        model = self.yolo_conv(model,filters=1024,ksize=3,name='conv_19')
        model = self.yolo_conv(model,filters=512,ksize=1,name='conv_20')
        model = self.yolo_conv(model,filters=1024,ksize=3,name='conv_21')
        model = self.yolo_conv(model,filters=512,ksize=1,name='conv_22')
        model = self.yolo_conv(model,filters=1024,ksize=3,name='conv_23')

        model = self.yolo_conv(model,filter=(self.box_per_cell*(self.num_classes+5)),ksize=1,name='conv_24')

        return model

    def model_loss(self,predictions,labels):

        predictions = tf.reshape(predictions,[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,self.num_classes+5])

        bbox_coord = tf.reshape(predictions[:,:,:,:,:4],[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,4])
        bbox_conf = tf.reshape(predictions[:,:,:,:,4],[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,1])
        bbox_classes = tf.reshape(predictions[:,:,:,:,5:],[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,self.num_classes])

        boxes1 = tf.stack([(1.0/(1.0 + tf.exp(-1.0*bbox_coord[:,:,:,:,0]))+self.offset)/self.cell_size,(1.0/(1.0 + tf.exp(-1.0*bbox_coord[:,:,:,:,1]))+ tf.transpose(self.offset,(0,2,1,3)))/self.cell_size,tf.sqrt(tf.exp(bbox_coord[:,:,:,:,2])*np.reshape(self.anchor_box[:5],[1,1,1,5])/self.cell_size),tf.sqrt(tf.exp(bbox_coord[:,:,:,:,3])*np.reshape(self.anchor_box[5:],[1,1,1,5])/self.cell_size)])

        bbox_coor_trans = tf.transpose(boxes1,(1,2,3,4,0))
        bbox_conf = 1.0 / (1.0 + tf.exp(-1.0*bbox_conf))
        bbox_classes = tf.nn.softmax(bbox_classes)

        response = tf.reshape(labels[:,:,:,:,0],[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell])
        boxes = tf.reshape(labels[:,:,:,:,1:5],[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,4])
        classes = tf.reshape(labels[:,:,:,:,5:],[self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,self.num_classes])

        iou = self.calc_iou(bbox_coor_trans,boxes)
        best_box = tf.to_float(tf.equal(iou,tf.reduce_max(iou,axis=-1,keep_dims=True)))
        confs = tf.expand_dims(best_box*response,axis=4)

        conid = self.noobject_scale*(1-confs) + self.object_scale*confs
        cooid = self.coordinate_scale * confs
        proid = self.class_scale * confs

        coo_loss = cooid * tf.square(bbox_coor_trans - boxes)
        con_loss = conid * tf.square(bbox_conf - confs)
        pro_loss = proid * tf.square(bbox_classes - classes)

        loss = tf.concat([coo_loss,con_loss,pro_loss],axis=4)
        loss = tf.reduce_mean(tf.reduce_sum(loss,axis=[1,2,3,4]),name='loss')

        return loss

    def calc_iou(self,pred_boxes,gt_boxes):

        box1 = tf.square(pred_boxes[:,:,:,:,2:4])
        boxes1_square = box1[:,:,:,:,0] * box1[:,:,:,:,1]
        box = tf.stack([pred_boxes[:,:,:,:,0]-box1[:,:,:,:,0]*0.5,pred_boxes[:,:,:,:,1]-box1[:,:,:,:,1]*0.5,pred_boxes[:,:,:,:,0]+box1[:,:,:,:,0]*0.5,pred_boxes[:,:,:,:,1]+box1[:,:,:,:,1]*0.5])
        boxes1 = tf.transpose(box,(1,2,3,4,0))

        box2 = tf.square(gt_boxes[:,:,:,:,2:4])
        boxes2_square = box2[:,:,:,:,0] * box2[:,:,:,:,0]
        box = tf.stack([gt_boxes[:,:,:,:,0]-box2[:,:,:,:,0]*0.5,gt_boxes[:,:,:,:,1]-box2[:,:,:,:,1]*0.5,gt_boxes[:,:,:,:,0]+box2[:,:,:,:,0]*0.5,gt_boxes[:,:,:,:,1]+box2[:,:,:,:,1]*0.5])
        boxes2 = tf.transpose(box,(1,2,3,4,0))

        left_up = tf.maximum(boxes1[:,:,:,:,:2],boxes2[:,:,:,:,:2])
        right_down = tf.minimum(boxes1[:,:,:,:,2:],boxes2[:,:,:,:,2:])

        intersection = tf.maximum(right_down - left_up,0.0)
        inter_square = intersection[:,:,:,:,0] * intersection[:,:,:,:,1]
        union_square = boxes1_square + boxes2_square - inter_square
        
        return tf.clip_by_value(1.0*inter_square/union_square,0.0,1.0)






