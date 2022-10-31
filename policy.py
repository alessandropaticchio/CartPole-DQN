import random
import torch
import math


def select_action(state, policy_net, n_actions, device, eps_end, eps_start, eps_decay, steps_done):

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
