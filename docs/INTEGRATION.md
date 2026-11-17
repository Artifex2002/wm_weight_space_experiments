# DINO-WM adapter contract

The Wall upgrade implements a hook for the pinned Meta DINO-WM baseline through
upstream_patch.py and cem_hook.py. See WALL_UPGRADE.md for commands. The CPU
quadratic-objective test executes the actual pinned CEM class; real Wall model
inference is still pending. The following generic contract explains the hook
and remains useful when porting to other planners. Freeze this implementation with the source packet before running
generation. Do not silently change the source after generating weights.

```python
from wsp_stopper.core import Features, Stopper
import json

stopper = Stopper(json.load(open("runs/trial-001/weights.json")))
# At the START of EACH solve, for EACH live environment instance:
features = Features(initial_sigma)   # standard deviation, not variance

# Inside that instance's original CEM loop, after the update:
x = features.update(
    candidate_costs,       # original scores, shape [population]
    elite_indices,        # original selected indices, shape [elites]
    updated_sigma,        # [horizon, internal_action_dim]
    original_returned_plan[0], # exactly the first action if returned NOW
)
if stopper.stop(x, features.k):
    # Exit only this instance's refinement; preserve its returned plan.
    # Pass that plan through the original inverse normalization and MPC execution.
    pass
```

The pass above is an integration placeholder, not an executable planner. Replace
it with the planner's actual break/freeze path. For GPU tensors, explicitly detach
and transfer only small statistics/arrays to CPU; count this overhead. Avoid
transferring full rollouts. This initial CPU extractor takes all candidate costs;
a production GPU extractor can compute the same formulas on device.

For batches, maintain per-instance histories, active masks and independently
frozen return plans. Using one global break or continuing to update stopped
instances changes the experimental policy. Masked GPU lanes may still execute;
count actual work separately from logical saved iterations.

Find whether the planner returns best candidate, proposal mean, or another plan.
Do not substitute one for another when extracting the first action or stopping.
If sigma is stored as variance, take its square root exactly once. Flattened
packed actions must be reshaped consistently; identify the first executed action
in internal coordinates. Warm starts reset feature history but use the new
solve's actual initial sigma.

The supplied feature contract requires nonnegative costs and positive initial
standard deviations. If source scoring violates that requirement, redesign the
contract BEFORE generation instead of hiding offsets or changing objective signs.

The episode limit M and the mapping from planner units to simulator steps must
be resolved, with a finite upper bound. Count C across all solves and compute
utility S - 0.05*C/(30*M). No later success information enters within-solve
features. The learned stopper training implementation is a separate experiment.
