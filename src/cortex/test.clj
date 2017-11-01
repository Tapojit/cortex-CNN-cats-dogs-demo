(ns cortex.test
  (:require [cortex.util :as util]
            [cortex.nn.execute :as execute]
            [mikera.image.core :as i]
            [think.image.patch :as patch]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [mikera.image.filters :as filters]
            ))
(def files (filter #(.isFile %) (file-seq (io/file "cats-vs-dogs/test1"))))
(defn- indexed-data-label-seq
  [files]
  (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
       (map-indexed vector)))



(defn image-file->observation
  [image]
  {:labels ["test"]
   :data
   (patch/image->patch
    (-> image (i/load-image) ((filters/grayscale)) (i/resize 52 52))
    :datatype :float
    :colorspace :gray)})
(def labels (map #(nth (nth % 1) 1) (indexed-data-label-seq files)))


(defn index->class-name
  [n]
  (nth ["dog" "cat"] n))
(defn predict
  [nippy]
  (let [obs (map #(image-file->observation (first (nth % 1))) (indexed-data-label-seq files))
        ]
    (map #(index->class-name (util/max-index (:labels %))) (execute/run nippy (into-array obs)))))
(def nippy
  (util/read-nippy-file "trained-network.nippy"))


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


;; (defn predicting
;;   []
;;   (predict nippy))
;; (image-by-index)

