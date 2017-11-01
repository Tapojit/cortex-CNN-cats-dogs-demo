(ns cortex.training
  (:require [clojure.java.io :as io]
            [mikera.image.core :as imagez]
            [cortex.experiment.util :as experiment-util]
            [cortex.nn.layers :as layers]
            [think.image.patch :as patch]
            [cortex.experiment.classification :as classification]))

(def data-folder "cats-vs-dogs/")
(def train (str data-folder "train1"))

(def validation (str data-folder "valid1"))
(def categories
  (into [] (map #(.getName %) (.listFiles (io/file train)))))
(def class-count
  (count categories))
(def class-mapping
  {:class-name->index (zipmap categories (range))
   :index->class-index (zipmap (range) categories)})


(def first-test-pic
  (->> (io/file train)
       (file-seq)
       (filter #(.isFile %))
       (first)))

(def image-width (.getWidth (imagez/load-image first-test-pic)))
image-width
(def train-ds
  (experiment-util/infinite-class-balanced-dataset (experiment-util/create-dataset-from-folder train class-mapping :image-aug-fn (:image-aug-fn {}))))
(def valid-ds
  (experiment-util/create-dataset-from-folder validation class-mapping ))

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
(defn- observation->image
  [observation]
  (patch/patch->image observation image-width))



(defn final-train
  []
  (let [listener (classification/create-listener observation->image
                                               class-mapping
                                               {})]
    (classification/perform-experiment
     (initial-description image-width image-width class-count)
     train-ds
     valid-ds
     listener)))



;; (into [] (map #(subs (.getName %) 0 3) (.listFiles (io/file (str data-folder "train")))))












