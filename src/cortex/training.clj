;;These are the libraries we need:
(ns cortex.training
  (:require [clojure.java.io :as io]
            [mikera.image.core :as imagez]
            [cortex.experiment.util :as experiment-util]
            [cortex.nn.layers :as layers]
            [think.image.patch :as patch]
            [cortex.experiment.train :as train]
            [cortex.nn.network :as network]
            [cortex.experiment.classification :as classification]))

;;Storing data directory
(def data-folder "cats-vs-dogs/")

;;Storing training data directory
(def train (str data-folder "train1"))

;;Storing validation data directory
(def validation (str data-folder "valid1"))


;;Storing vector of class names(from folder names in training directory).
(def categories
  (into [] (map #(.getName %) (.listFiles (io/file train)))))

;;Storing number of classes
(def class-count
  (count categories))

;;Storing a mapping from class name to index and index to class name
(def class-mapping
  {:class-name->index (zipmap categories (range))
   :index->class-index (zipmap (range) categories)})

;;An example preprocessed image
(def first-test-pic
  (->> (io/file train)
       (file-seq)
       (filter #(.isFile %))
       (first)))

;;Storing width of example image
(def image-width (.getWidth (imagez/load-image first-test-pic)))

;;Creating and storing training dataset for use by CNN architecture
(def train-ds
  (experiment-util/infinite-class-balanced-dataset (experiment-util/create-dataset-from-folder train class-mapping :image-aug-fn (:image-aug-fn {}))))

;;Creating and storing validation dataset for use by CNN architecture
(def valid-ds
  (experiment-util/create-dataset-from-folder validation class-mapping ))

;;IMPORTANT!
;;CNN architecture is being defined here.
;;Takes in: i) image width ii) image height iii) number of classes
(defn initial-description
  [w h classes]
  [(layers/input w h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/relu)
   (layers/max-pooling 2 0 2)
   (layers/dropout 0.9)
   (layers/convolutional 5 0 1 50)
   (layers/relu)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 1024)
   (layers/relu :center-loss {:label-indexes {:stream :labels}
                              :label-inverse-counts {:stream :labels}
                              :labels {:stream :labels}
                              :alpha 0.9
                              :lambda 1e-4})
   (layers/dropout 0.5)
   (layers/linear classes)
   (layers/softmax :id :labels)])

;;Note-In the last RELU layer, :lambda is the learning rate, which is set at 0.0001.


;;Converts the network into a computational graph
(def net (network/linear-network (initial-description image-width image-width class-count)))

;;This function trains the network
(defn final-train
  "Takes in size of individual batch of images the network is trained on per epoch and number of epochs the network
  is trained over"
  [batch-size epoch-no]
  (train/train-n net  train-ds valid-ds :batch-size batch-size :epoch-count epoch-no))










