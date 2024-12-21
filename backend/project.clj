(defproject backend "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [com.openai/openai-java "0.8.0"]
                 [yogthos/config "1.2.1"]
                 [clojure-lanterna "0.9.7"]]
  :main ^:skip-aot backend.core
  :repl-options {:init-ns backend.core})
