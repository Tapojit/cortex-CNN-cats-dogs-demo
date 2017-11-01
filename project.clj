(defproject cortex "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [clj-time "0.9.0"]
                 [net.mikera/imagez "0.12.0"]
                 [thinktopic/think.image "0.4.12"]
                 [thinktopic/cortex "0.9.11"]
                 [thinktopic/experiment "0.9.11"]
                 ]
  :jvm-opts ["-Xmx2000m"]
  :uberjar-name "classify-example.jar"
  :main ^:skip-aot cortex.core
  :clean-targets ^{:protect false} [:target-path
                                    "figwheel_server.log"
                                    "resources/public/out/"
                                    "resources/public/js/app.js"])
