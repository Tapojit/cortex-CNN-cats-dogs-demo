;;Libraries required for testing our CNN
(ns cortex.test
  (:require [cortex.util :as util]
            [cortex.nn.execute :as execute]
            [mikera.image.core :as i]
            [think.image.patch :as patch]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [mikera.image.filters :as filters]
            ))

;;Gets a collection of all images in test1 directory
(def files (filter #(.isFile %) (file-seq (io/file "cats-vs-dogs/test1"))))

;;For a given collection of files, returns a collection of [index [file label]] representing individual files
(defn- indexed-data-label-seq
  [files]
  (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
       (map-indexed vector)))


;;Converts a given image to an observation format(so it can be interpreted by the CNN)
(defn image-file->observation
  [image]
  {:labels ["test"]
   :data
   (patch/image->patch
    (-> image (i/load-image) ((filters/grayscale)) (i/resize 52 52))
    :datatype :float
    :colorspace :gray)})

;;A collection of true labels of test images
(def labels (map #(nth (nth % 1) 1) (indexed-data-label-seq files)))

;;A mapping of index to class name, where 0->"dog" and 1->"cat"
(defn index->class-name
  [n]
  (nth ["dog" "cat"] n))

;;Given a nippy file(which contains trained weights), makes predictions on test images
;;and returns a collection of predicted labels
(defn predict
  [nippy]
  (let [obs (map #(image-file->observation (first (nth % 1))) (indexed-data-label-seq files))
        ]
    (map #(index->class-name (util/max-index (:labels %))) (execute/run nippy (into-array obs)))))

;;Reading nippy file containing trained weights
(def nippy
  (util/read-nippy-file "trained-network.nippy"))


;;Iterating over true and predicted labels to calculate accuracy score in
;;predicting the test dataset.
(defn accuracy
  []
  (let [predicted (predict nippy)
        actual labels]
    (loop [idx 0
           correct 0]
      (if (== (- (count actual) 1) idx)
        (float (/ correct (count actual)))
        (recur (if (= (nth predicted idx) (nth actual idx))
                 (inc correct)
                 correct
                 )
               (inc idx))
        )

      )
    ))



