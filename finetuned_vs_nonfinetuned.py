#!/usr/bin/env python
# coding: utf-8

# # Compare finetuned vs. non-finetuned models

# In[ ]:


import os
import lamini

lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")


# In[ ]:


from llama import BasicModelRunner


# ### Try Non-Finetuned models

# In[ ]:


non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-hf")


# In[ ]:


non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")


# In[ ]:


print(non_finetuned_output)


# In[ ]:


print(non_finetuned("What do you think of Mars?"))


# In[ ]:


print(non_finetuned("taylor swift's best friend"))


# In[ ]:


print(non_finetuned("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))


# ### Compare to finetuned models 

# In[ ]:


finetuned_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")


# In[ ]:


finetuned_output = finetuned_model("Tell me how to train my dog to sit")


# In[ ]:


print(finetuned_output)


# In[ ]:


print(finetuned_model("[INST]Tell me how to train my dog to sit[/INST]"))


# In[ ]:


print(non_finetuned("[INST]Tell me how to train my dog to sit[/INST]"))


# In[ ]:


print(finetuned_model("What do you think of Mars?"))


# In[ ]:


print(finetuned_model("taylor swift's best friend"))


# In[ ]:


print(finetuned_model("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))


# In[ ]:




