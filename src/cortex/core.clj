(ns cortex.core
  (:require
;;    [cortex.prepare :as prep]
   [cortex.training :as train]
;;    [cortex.test :as test]

            )
  (:gen-class))
(defn -main
  [& args]
  (do
;;     (prep/build-image-data "cats-vs-dogs/train" "cats-vs-dogs/train1" "cats-vs-dogs/valid1" "cats-vs-dogs/test1" 52)
    (train/final-train)
;;     (println (str "Test accuracy: " (test/accuracy)))
  ))





