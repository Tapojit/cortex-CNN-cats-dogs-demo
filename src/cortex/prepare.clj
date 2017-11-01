(ns cortex.prepare
  (:require [clojure.java.io :as io]
            [mikera.image.filters :as filters]
            [mikera.image.core :as imagez]
            [clojure.string :as string]))

(defn- gather-files
  [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))

(defn- indexed-data-label-seq
  [files]
  (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
       (map-indexed vector)))

(defn img-preprocess
  [output-dir image-size [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png")]
    (when-not (.exists (io/file img-path))
      (println "> " img-path)
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          ((filters/grayscale))
          (imagez/resize image-size image-size)
          (imagez/save img-path)))))


(defn img-preprocess-test
  [output-dir image-size [idx [file label]]]
  (let [img-path (str output-dir "/" label "." idx ".png")]
    (when-not (.exists (io/file img-path))
      (println "> " img-path)
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          ((filters/grayscale))
          (imagez/resize image-size image-size)
          (imagez/save img-path)))))

(defn build-image-data
  [original-dir training-dir valid-dir test-dir img-size]
  (let [files (gather-files original-dir)
        pfiles (partition (int (/ (count files) 5)) (shuffle files))
        test-labels (indexed-data-label-seq (first pfiles))
        valid-labels (indexed-data-label-seq (first (rest pfiles)))
        training-labels (indexed-data-label-seq (apply concat (rest (rest pfiles))))
        train-fn (partial img-preprocess training-dir img-size)
        valid-fn (partial img-preprocess valid-dir img-size)
        test-fn (partial img-preprocess-test test-dir img-size)]
    (dorun (pmap train-fn training-labels))
    (dorun (pmap valid-fn valid-labels))
    (dorun (pmap test-fn test-labels))))






