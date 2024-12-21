(ns backend.core
  (:require [config.core :refer [env]])
  (:import
    [com.openai.client.okhttp OpenAIOkHttpClient]
    (com.openai.models ChatCompletionCreateParams
                       ChatCompletionMessageParam ChatCompletionUserMessageParam
                       ChatCompletionUserMessageParam$Content
                       ChatCompletionUserMessageParam$Role
                       ChatModel))
  (:gen-class))

;; Initialize the OpenAI client
(defn initialize-client []
  (-> (OpenAIOkHttpClient/builder)
      (.apiKey (env :openai-api-key))
      (.build)))

;; Function to send a message to OpenAI
(defn send-message
  "Sends a message to the OpenAI API and returns the response."
  [client prompt]

  (let [params (-> (ChatCompletionCreateParams/builder)
                   (.messages [(-> (ChatCompletionMessageParam/ofChatCompletionUserMessageParam
                                    (-> (ChatCompletionUserMessageParam/builder)
                                        (.role ChatCompletionUserMessageParam$Role/USER)
                                        (.content (ChatCompletionUserMessageParam$Content/ofTextContent prompt))
                                        .build))
                                  )])
                   (.model ChatModel/GPT_4)
                   (.build))]

    (-> client
        (.chat)
        (.completions)
        (.create params))))

(defn -main
  "Simple TUI for chatting with OpenAI"
  [& args]
  (println "Welcome to Wrighter Chat! Type 'exit' to quit.")
  (let [client (initialize-client)]
    (loop []
      (print "You: ")
      (flush)
      (let [user-input (read-line)]
        (if (= "exit" (clojure.string/lower-case user-input))
          (println "Goodbye!")
          (do
            (let [response (send-message client user-input)]
              (println "Assistant:" (-> response
                                        .choices
                                        first
                                        .message
                                        .content
                                        .get)))
            (recur)))))))