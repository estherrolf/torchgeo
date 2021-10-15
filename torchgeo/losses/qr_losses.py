import torch as T


def loss_on_prior_simple(logged_output, 
                         target):
    # inputs: assumes q and r are both valid probablity dists over the c channel
    # logged NN output = log(q)
    # target = prior
   
    q = T.exp(logged_output)
    q_bar = q.mean(axis=(0,2,3))
    qbar_log_S = (q_bar * T.log(q_bar) ).sum()
        
    q_log_p = T.einsum('bcxy,bcxy->bxy', q, T.log(target)).mean()

    loss = qbar_log_S - q_log_p
    return loss
   
def loss_on_prior_reversed_kl_simple(logged_output, 
                                      target):
    # inputs: assumes q and r are both valid probablity dists over the c channel
    # logged NN output = log(q)
    # target = prior

    q = T.exp(logged_output)

    # they're in batches 
    z = T.nn.functional.normalize(q,p=1,dim=(0,2,3))
    r = T.nn.functional.normalize(z * target, p=1,dim=1)
        
    loss = T.einsum('bcxy,bcxy->bxy', r, T.log(r) - T.log(q)).mean() 
    return loss