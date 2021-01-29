### Where is this deployed

It is deployed as an HTTP-only endpoint on AWS EKS (Kubernetes service from AWS)

Host(HTTP-only): **`api-audio.huggingface.co`** => AWS ELB => EKS

SSL termination in EKS made my brain hurt so I'm simply proxying and handling SSL termination in nginx on banana.

Host(HTTPS): **`api-audio-frontend.huggingface.co`**

