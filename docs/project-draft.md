# Blindfolded Rubik's Cube Solving Through Tactile Robotic Manipulation

## Executive Summary

### Project objective

Develop a robotic manipulation system capable of inspecting a Rubik's Cube visually, memorizing its configuration, and subsequently solving it without access to visual observations.

After the inspection phase, the robot must rely exclusively on:

- Tactile sensor measurements.
- Joint positions, velocities, and available actuator feedback.
- Internal state estimates.
- Information retained from the visual inspection.
- Previously executed actions and planned actions.

The system will be developed entirely in simulation, using MuJoCo, a dexterous robotic hand, and reinforcement learning.

The proposed implementation uses a Sharpa Wave robotic hand mounted on a robotic arm, although the initial manipulation experiments may use a fixed wrist or a simplified robotic configuration.

The project explores a fundamental robotics question:

**Can an embodied agent maintain an accurate model of an articulated object and successfully manipulate it over long sequences without direct visual observation?**

The ultimate demonstration is blindfolded Rubik's Cube solving, but the underlying research concerns tactile state estimation, manipulation under partial observability, and the ability to detect and recover from physical execution errors.

### Recommended research strategy

The system should not initially be trained end-to-end to solve Rubik's Cubes using reinforcement learning.

Instead, develop a hierarchical architecture comprising:

1. A visual inspection system.
2. An explicit symbolic representation of the cube.
3. A conventional Rubik's Cube solving algorithm.
4. A learned dexterous manipulation controller.
5. A tactile/proprioceptive state estimator.
6. An execution monitor capable of detecting manipulation errors.
7. A recovery and replanning mechanism.

The central machine-learning contribution should be the **tactile manipulation policy and the state-estimation/world-model component**, rather than rediscovering a symbolic algorithm for solving Rubik's Cubes.

### Scope recommendation

Begin with a **2×2×2 Rubik's Cube**, not a conventional 3×3×3 cube.

The 2×2×2 cube provides a substantially simpler starting point:

- Eight movable corner pieces.
- No edge pieces.
- Fewer possible configurations.
- Simpler state representation.
- Easier integration with existing research.
- Shorter solution sequences.
- Direct relevance to the 2026 FINGR results.

The 3×3×3 cube should be treated as an advanced extension.

### Expected outcomes

A successful project should produce:

- A functioning MuJoCo environment with a dexterous hand and articulated cube.
- Validated simulated tactile sensing.
- A policy capable of executing individual cube-layer turns.
- A tactile state estimator capable of tracking object and layer motion.
- A blindfolded controller capable of executing multiple turns.
- A structured evaluation comparing different sensing and estimation architectures.
- Ideally, a demonstration of blindfolded 2×2×2 cube solving.

A complete arbitrary-scramble solve should be considered a stretch goal, not the minimum requirement for project success.

---

# Part I — Research Background

## 1. Relevant existing research

Several research projects provide foundations for the proposed system.

### 1.1 OpenAI Dactyl — Solving Rubik's Cube (2019)

OpenAI demonstrated that a dexterous Shadow Hand could execute Rubik's Cube manipulation through policies trained in simulation.

The system used a combination of reinforcement learning, automatic domain randomization, physical manipulation policies, and symbolic cube-solving logic.

The work established that high-dimensional robotic-hand control could be learned for complex manipulation sequences.

However, the system depended on external visual information and specialized cube-state sensing.

**Relevance:** Provides precedent for hierarchical Rubik's Cube manipulation and simulation-trained dexterous policies.

**Main limitation for this project:** It does not establish purely tactile execution after visual inspection.

Reference:

OpenAI, *Solving Rubik's Cube with a Robot Hand* (2019).

https://openai.com/index/solving-rubiks-cube/

### 1.2 Learning Purely Tactile In-Hand Manipulation (2022)

Sievers, Pitz, and Bäuml demonstrated continuous cube rotation using only joint-position and torque feedback.

Their system operated a DLR Hand-II without visual observations.

The policy was trained in simulation using Soft Actor-Critic and transferred to a physical robot.

The work demonstrated that continuous dexterous regrasping and object rotation can be achieved without external object tracking.

The authors reported approximately 600 CPU-hours of training for their system.

**Relevance:** Strong evidence that a learned controller can exploit proprioceptive and implicit tactile information to maintain and manipulate a grasped object.

**Limitation:** The cube was treated as a rigid object. The task did not require independently rotating articulated layers or solving a puzzle.

Reference:

Sievers et al. (2022), *Learning Purely Tactile In-Hand Manipulation with a Torque-Controlled Hand*.

https://arxiv.org/abs/2204.03698

### 1.3 Dextrous Tactile In-Hand Manipulation Using a Modular Reinforcement Learning Architecture (2023)

Pitz et al. combined a learned manipulation policy with a differentiable particle-filter state estimator.

The system reoriented a rigid cube among 24 discrete target orientations without external visual observations.

The authors reported a 92% success rate in simulation under significant domain randomization.

A particularly relevant design decision was the separation between:

- A manipulation policy.
- An object-state estimator.

During policy training, privileged simulator state could be used to simplify learning. During deployment, the learned estimator replaced privileged object-pose measurements.

**Relevance:** This is one of the strongest architectural references for the proposed project.

It demonstrates that a tactile state estimator and manipulation policy can be developed separately and subsequently integrated.

**Limitation:** Reorienting a rigid cube is not equivalent to manipulating a Rubik's Cube's internal layers.

Reference:

Pitz et al. (2023), *Dextrous Tactile In-Hand Manipulation Using a Modular Reinforcement Learning Architecture*.

https://arxiv.org/abs/2303.04705

### 1.4 Rotating without Seeing: Towards In-Hand Dexterity through Touch (2023)

Yin et al. investigated in-hand rotation using tactile feedback without visual observations.

Their system used an Allegro Hand with distributed binary contact sensors.

An important finding was that coarse, spatially distributed contact measurements could support dexterous manipulation.

The system used reinforcement learning and demonstrated transfer from simulation to real hardware.

**Relevance:** Motivates comparing simple contact sensing with high-resolution fingertip taxels.

It also suggests that expanding tactile coverage beyond the fingertips may be more important than simply increasing fingertip resolution.

Reference:

Yin et al. (2023), *Rotating without Seeing: Towards In-hand Dexterity through Touch*.

https://arxiv.org/abs/2303.10880

### 1.5 FINGR — Learning Dexterous Hand Control for Real-World Rubik's Cube Solving (2026)

This is the most directly relevant recent work.

FINGR demonstrates robotic manipulation of a physical 2×2×2 Rubik's Cube.

Its architecture combines:

- A dexterous robotic hand.
- A seven-joint robotic arm.
- Tactile measurements.
- Finger-relative geometric representations.
- Predictive supervision of future interactions.
- A learned flow-based action policy.
- Symbolic cube solving.
- Table-assisted regrasping.

The learned policy controls 13 joints across three fingers while the other fingers stabilize the cube.

The authors reported approximately 99% success across 300 individual layer-turn attempts and successfully solved all ten tested scrambled 2×2×2 cubes.

One especially useful contribution is the prediction of future contact changes, joint motion, and layer-turn progress.

**Relevance:** Demonstrates that reliable cube-layer turning can be learned and integrated into a complete robotic solving system.

**Critical difference:** FINGR receives continuously updated geometric information about the cube, including articulated geometry and turn-progress information.

Your project should explicitly remove this information after the initial inspection phase.

The proposed contribution is therefore not simply to reproduce FINGR, but to investigate how much of its sensing and feedback functionality can be replaced by tactile-proprioceptive state estimation.

Reference:

Liang et al. (2026), *FINGR: Learning Dexterous Hand Control for Real-World Rubik's Cube Solving*.

https://arxiv.org/abs/2609.33973

### 1.6 DexTouch-WM — Action-Conditioned Tactile World Models (2026)

DexTouch-WM explores learning predictive representations of tactile interactions conditioned on actions.

Its approach combines tactile and visual information and investigates transferring contact-related representations between embodiments.

The project is relevant to developing a learned model of how tactile observations evolve during manipulation.

However, predicting tactile observations does not automatically provide a reliable estimate of Rubik's Cube state.

The proposed project should use tactile prediction as an auxiliary objective alongside explicit physical-state estimation.

Reference:

Qin et al. (2026), *DexTouch-WM: Learning Action-Conditioned Tactile World Models from Human Touch for Dexterous Robot Manipulation*.

https://arxiv.org/abs/2609.20649

---

## 2. Research contribution and novelty

The project should be framed carefully.

Existing research already demonstrates:

- Rubik's Cube manipulation with learned dexterous policies.
- Tactile-only rigid-object manipulation.
- Tactile state estimation.
- Robotic 2×2×2 cube solving.
- Learned predictive representations of tactile interactions.

Consequently, none of these elements individually establishes novelty.

The proposed research lies in their combination under a specific constraint:

**The robot may visually inspect an articulated cube before execution but cannot access visual or privileged object-state observations while solving it.**

Potential research contributions include:

1. A benchmark for blindfolded articulated-object manipulation.
2. A tactile estimator of Rubik's Cube layer motion.
3. A controller capable of detecting incomplete or incorrect turns.
4. A hierarchical architecture combining symbolic memory with learned physical-state estimation.
5. An investigation into the relationship between tactile resolution and long-horizon manipulation reliability.
6. A learned action-conditioned tactile world model for manipulation monitoring and recovery.

Novelty should be verified through a more exhaustive literature review before making publication claims.

---

# Part II — Formal Problem Definition

## 3. Task specification

The task consists of two phases.

### Phase A — Visual inspection

The robot begins with a scrambled Rubik's Cube.

It may use cameras and physical manipulation to inspect the cube.

The system must reconstruct the complete symbolic configuration before the camera is disabled.

The inspection phase ends when the system commits to an internal representation of the cube.

### Phase B — Blindfolded execution

All camera observations are disabled.

The robot must solve the cube using the memorized configuration and nonvisual feedback.

The system may use:

- Current and historical tactile measurements.
- Current and historical robot proprioception.
- Internal recurrent state.
- The stored symbolic configuration.
- The stored manipulation plan.
- Predictions made by learned models.
- Previous actions and their observed consequences.

It must not receive:

- Updated camera frames.
- Ground-truth cube pose.
- Ground-truth cubie poses.
- Ground-truth layer angles.
- Hidden simulator contact states not represented by the defined tactile sensor interface.
- The simulator's true post-action symbolic cube configuration.

The last restriction is especially important.

A policy that receives exact cube joint angles after vision is disabled is not solving the intended state-estimation problem.

### Evaluation

The simulator may access complete ground-truth information to calculate metrics, rewards during training, and final success.

However, evaluation instrumentation must remain separate from information provided to the deployed agent.

---

## 4. Mathematical formulation

The complete physical state can be represented as:

\[
x_t =
(q_t,\dot q_t,p_t,R_t,\theta_t,\dot\theta_t,c_t)
\]

where:

- \(q_t\): robot joint positions.
- \(\dot q_t\): robot joint velocities.
- \(p_t\): cube position.
- \(R_t\): cube orientation.
- \(\theta_t\): physical cube-layer configuration.
- \(\dot\theta_t\): layer angular velocities.
- \(c_t\): contact configuration and related physical state.

The symbolic puzzle state is:

\[
S_t =
(\sigma_t,\omega_t)
\]

where:

- \(\sigma_t\): permutation of cube pieces.
- \(\omega_t\): orientation of cube pieces.

For a 2×2×2 cube, only corner permutations and corner orientations are needed.

These symbolic variables must not be confused with physical joint angles.

The observation available after vision is disabled is:

\[
o_t =
(q_t,\dot q_t,\tau_t,T_t,a_{t-1})
\]

where:

- \(\tau_t\): available joint-effort or actuator-feedback measurements.
- \(T_t\): tactile measurements.
- \(a_{t-1}\): previous control action.

The system must estimate the hidden physical state:

\[
\hat x_t = f_\phi(o_{0:t},a_{0:t-1})
\]

It must also maintain a belief over the symbolic cube state:

\[
b_t(S)=P(S_t=S\mid o_{0:t},a_{0:t-1},S_0)
\]

The objective is to maximize the probability of reaching the solved configuration while maintaining a stable grasp.

\[
\max_\pi
P(S_T=S_{\text{solved}})
\]

subject to:

\[
o_t \not\supseteq
\{\text{vision},\text{privileged cube state}\}
\]

during the blindfolded phase.

This is naturally a partially observable control problem.

---

## 5. Two different kinds of memory

The project should explicitly distinguish two representations.

### 5.1 Symbolic memory

Stores the puzzle's logical configuration.

For example:

- Corner permutation.
- Corner orientation.
- Planned solution moves.
- Which moves have been confirmed.
- Uncertainty about possibly failed moves.

This representation is discrete.

### 5.2 Physical-state memory

Stores the estimated physical configuration of the robot and cube.

For example:

- Cube orientation in the robot-hand frame.
- Position of the cube relative to fingertips.
- Current layer angle.
- Contact locations.
- Estimated sliding motion.
- Grasp stability.

This representation is continuous.

The two interact but serve different purposes.

A whole-cube rotation changes the cube's pose relative to the robot but does not change its internal scrambled configuration.

A successful face turn changes the internal puzzle state.

The system must distinguish these cases.

---

# Part III — Simulation Environment

## 6. Technology stack

### Recommended initial stack

| Component | Technology |
|---|---|
| Physics simulation | MuJoCo |
| Robot hand | Sharpa Wave |
| Robot arm | Franka Panda, or similar |
| Tactile observations | MuJoCo contact/tactile sensing |
| Environment interface | Gymnasium |
| Machine learning | PyTorch |
| Initial RL algorithm | SAC or PPO |
| State estimator | GRU, transformer, or particle filter |
| Symbolic solver | Existing 2×2×2 solver |
| Experiment tracking | TensorBoard or Weights & Biases |
| Configuration management | YAML/Hydra |
| Evaluation | Custom deterministic benchmark |
| Visualization | MuJoCo viewer and recorded video |

The initial implementation should prioritize standard CPU-based MuJoCo.

GPU acceleration should be considered after establishing a working simulator and sensor interface.

### 6.1 Sharpa Wave model

MuJoCo Menagerie provides a Sharpa Wave model with:

- Five fingers.
- 22 actuated joints.
- Position actuators.
- Fingertip collision geometries.
- Compliant-contact approximations.
- A wrist mounting site.

The model is a suitable starting point.

However, its visual representation of tactile pads does not constitute a calibrated tactile sensing model.

Sensor integration will be necessary.

### 6.2 Robot arm

A seven-degree-of-freedom arm is appropriate.

The Franka Panda model is a reasonable choice because it is widely used in robotics simulation.

A robotic arm can provide:

- Cube acquisition.
- Wrist orientation.
- Gross positioning.
- Stabilization.
- Table-assisted regrasping.

However, it should initially be treated as a separate subsystem from finger manipulation.

A fixed-wrist experiment is substantially easier to debug.

### 6.3 Cube model

The cube must support physical layer rotation, not merely visual animation.

Required properties include:

- Rigid cubies or an equivalent articulated geometry.
- Correct exposed collision surfaces.
- Physically meaningful relative layer motion.
- Joint constraints or other mechanisms permitting legal rotations.
- Stable contacts with fingertips.
- Friction and contact dynamics.
- A well-defined mapping between physical motion and symbolic cube moves.

The historical OpenAI RoboGym repository includes several relevant cube models, including configurations with two or six internal degrees of freedom.

These should be studied as implementation references rather than assumed to be immediately compatible with modern MuJoCo.

### Important modeling issue

A Rubik's Cube is mechanically more complicated than an ordinary articulated robot.

Its connectivity effectively changes when different faces turn.

A conventional fixed kinematic tree is not a complete representation of arbitrary cube motion.

Historical research environments may use specialized joint arrangements and abstractions.

The project should therefore begin by testing whether the selected cube representation can execute arbitrary legal move sequences without:

- Cubie interpenetration.
- Loss of structural integrity.
- Constraint instability.
- Unexpected collisions.
- Illegal transitions between cube states.

A simplified model may be sufficient for learning layer-turn mechanics, but it must preserve the physical challenges being studied.

---

## 7. Recommended cube-modeling strategy

Use three increasingly sophisticated models.

### Model A — Single-axis articulated cube

The first environment contains a cube with one rotatable layer.

Only one face-turn direction or axis is enabled.

Advantages:

- Minimal physics complexity.
- Easy debugging.
- Simple rewards.
- Clear tactile signals.
- Easy measurement of angle-estimation accuracy.

This environment is intended for learning a single physical skill, not for solving complete puzzles.

### Model B — Restricted multi-axis cube

Enable two distinct face-turn operations.

This allows the policy to learn:

- Different finger-contact patterns.
- Switching between layer-turn actions.
- Restabilizing the cube.
- Successive face turns.

This is the first appropriate environment for sequence-level evaluation.

### Model C — Complete 2×2×2 cube

Support arbitrary legal face-turn sequences.

The model must maintain a valid correspondence between its physical configuration and the symbolic puzzle state.

This becomes the environment used for complete blindfolded solving.

The progression is important because debugging contact dynamics and learning an entire cube-solving task simultaneously would be inefficient.

---

# Part IV — Tactile Sensing

## 8. Sensor requirements

The tactile sensing system must provide enough information for the controller to infer:

1. Whether a fingertip is touching the cube.
2. How the contact changes during movement.
3. Whether a finger is supporting or pushing the cube.
4. Whether the contact is sliding.
5. Whether the selected layer is moving relative to the supported cube.
6. Whether a grasp is becoming unstable.

The sensor does not need to recognize sticker colors.

The initial visual inspection provides that information.

---

## 9. Sensor configurations

Three configurations should be implemented.

### Configuration A — Contact-only

Each fingertip provides a binary contact indicator:

\[
T_i\in\{0,1\}
\]

This is the simplest tactile baseline.

It is computationally inexpensive and useful for verifying the sensing pipeline.

It also provides an important experimental comparison against higher-dimensional tactile sensing.

### Configuration B — Fingertip force measurements

Each fingertip provides a small force representation:

\[
T_i=[F_x,F_y,F_z]
\]

These values may be obtained from appropriate force sensing or derived from simulated contact forces.

This configuration provides information about loading and contact changes.

It does not directly provide detailed spatial contact location.

### Configuration C — Distributed tactile taxels

Each fingertip provides a spatial array of tactile values:

\[
T_i\in\mathbb{R}^{H\times W\times C}
\]

For example, with an 8×8 sampling layout and three channels:

\[
T_i\in\mathbb{R}^{8\times8\times3}
\]

Across five fingertips:

\[
T\in\mathbb{R}^{5\times8\times8\times3}
\]

The complete tactile observation contains 960 values.

This size is manageable for a small neural encoder.

### MuJoCo-specific considerations

MuJoCo's native tactile sensor samples penetration depth and magnitudes of tangential sliding velocities at sensor-mesh vertices.

These are not direct measurements of signed tangential force.

In particular, the reported tangential-speed magnitudes do not, on their own, preserve complete slip-direction information.

The exact sensor layout and number of taxels depend on the selected sensor mesh.

Consequently, tactile measurements should be validated against known physical motions before training the policy.

---

## 10. Sensor validation experiments

Before learning manipulation, construct a standalone sensor test environment.

### Experiment T1 — Static contact

Press a cube against a fingertip.

Verify:

- Contact appears in the expected region.
- Sensor values increase or change with contact depth.
- Measurements return to zero after release.

### Experiment T2 — Sliding contact

Slide the cube tangentially across the fingertip.

Verify that the sensor captures changes consistent with sliding motion.

### Experiment T3 — Contact location

Apply contact at several different positions on the fingertip.

Check whether the spatial tactile representation distinguishes the positions.

### Experiment T4 — Multiple contacts

Apply multiple simultaneous contacts.

Determine whether the sensor representation preserves enough information to distinguish them.

### Experiment T5 — Noise and perturbations

Introduce controlled noise, latency, and sensor dropout.

Evaluate the robustness of the tactile representation.

### Acceptance criteria

At the end of this phase, the system should demonstrate:

- Repeatable contact detection.
- Consistent measurements across repeated physical events.
- Meaningful responses to sliding.
- Correct association between measurements and fingertip geometry.
- A documented sensor interface usable by the learning system.

If the sensor cannot reliably distinguish the physical interactions needed for the task, the design should be revised before advancing to manipulation learning.

---

# Part V — Manipulation Control

## 11. Hierarchical control architecture

The proposed architecture contains several distinct modules.

### High-level planner

Receives the current estimated symbolic cube state.

Produces the next desired symbolic move.

Example:

\[
m_t=R'
\]

### Skill controller

Interprets the symbolic move as a physical manipulation objective.

For example:

- Stabilize the cube.
- Rotate the right layer counterclockwise by 90°.
- Return the fingers to a stable configuration.

### Low-level manipulation policy

Receives tactile/proprioceptive observations and a manipulation goal.

Produces actuator commands.

### State estimator

Estimates hidden physical quantities from sensor history.

### Execution monitor

Determines whether the commanded move:

- Completed successfully.
- Failed.
- Remains incomplete.
- Produced an unexpected result.

### Recovery controller

Attempts to restore a known, stable state when execution becomes uncertain.

This modular decomposition improves debugging and allows individual components to be trained and evaluated independently.

---

## 12. Action representation

The Sharpa Wave has 22 actuated joints.

With a seven-joint arm, the complete action space could contain 29 independently controlled dimensions.

However, controlling all 29 dimensions with a single RL policy should not be the first implementation.

### Recommended initial approach

Fix the wrist and train finger manipulation.

The action can be defined as:

\[
a_t\in[-1,1]^{22}
\]

Each element represents a bounded desired change in joint position.

The commanded target becomes:

\[
q_{t+1}^{target}
=
\operatorname{clip}
(q_t^{target}+k_a a_t,q_{min},q_{max})
\]

where \(k_a\) controls the maximum change in joint target.

Use position control with appropriate filtering and rate limits.

### Alternative: reduced action space

Following the general design principle demonstrated by FINGR, some fingers can maintain a support grasp while others perform the desired turn.

For example:

- Thumb and middle finger: stabilization.
- Index, ring, and little fingers: manipulation.

The exact assignment depends on which face is being turned and the geometry of the grasp.

A reduced action space can make initial learning more manageable.

However, it may also restrict the set of achievable turns.

The final system should support different finger roles or grasp configurations.

---

## 13. Learning an individual face turn

### Task definition

The robot begins while holding a cube in a stable pose.

A target layer and target angle are supplied.

The objective is to execute the rotation while preserving the grasp.

Example:

\[
\theta_{\text{target}}=\frac{\pi}{2}
\]

The controller should rotate the selected layer by 90° relative to the rest of the cube.

### Reward design

A possible reward is:

\[
r_t=
w_1r_{\text{progress}}
+w_2r_{\text{completion}}
+w_3r_{\text{stability}}
-w_4r_{\text{drop}}
-w_5r_{\text{unwanted}}
-w_6r_{\text{effort}}
\]

Components:

**Turn progress:** Reward reduction in angular distance to the target.

**Completion:** Reward reaching and maintaining the target angle.

**Grasp stability:** Encourage maintaining control of the cube.

**Drop penalty:** Penalize dropping or losing the object.

**Unwanted motion:** Penalize unintended cube-body motion or rotation of the wrong layer.

**Effort penalty:** Discourage excessive actuation and unstable finger movements.

The ground-truth target-layer angle may be used to compute rewards during training.

It must not be included directly in the blindfolded policy observations unless the value comes from the estimator.

### Curriculum

Use the following sequence:

1. Begin with the cube already grasped.
2. Train a single direction of rotation.
3. Train both clockwise and counterclockwise turns.
4. Randomize initial finger configurations.
5. Introduce variable contact friction.
6. Train turn completion and finger recovery.
7. Train multiple consecutive turns.
8. Introduce new layers and regrasping.

### Initial algorithm choice

Soft Actor-Critic is a strong candidate.

It has relevant precedent in tactile dexterous-manipulation research and can reuse previously collected simulation transitions.

PPO is also reasonable, particularly if many parallel simulation environments become available.

The best choice should be determined empirically.

### A practical alternative

If reinforcement learning from scratch is unstable, generate demonstrations using:

- Scripted motions.
- Trajectory optimization.
- Hand-designed grasp and turning sequences.
- Privileged-state expert policies.

Then use behavior cloning or reinforcement-learning fine-tuning.

This is particularly attractive because the project's central scientific question concerns blindfolded execution and state estimation, not whether useful finger motions can be discovered without any prior structure.

---

# Part VI — Tactile State Estimation

## 14. State-estimation objectives

The estimator should infer physical quantities that cannot be measured directly after the camera turns off.

At minimum:

\[
\hat x_t=
(\hat p_t,\hat R_t,\hat\theta_t,\hat c_t)
\]

where:

- \(\hat p_t\): cube position.
- \(\hat R_t\): whole-cube orientation.
- \(\hat\theta_t\): relevant layer angle.
- \(\hat c_t\): estimated contact state.

It should also predict uncertainty.

The estimator must distinguish between:

- A completed face turn.
- An incomplete face turn.
- A failed turn with little or no layer motion.
- A whole-cube rotation.
- An unexpected turn of another layer.

This distinction is the foundation of reliable blindfolded solving.

---

## 15. Baseline estimator: recurrent neural network

A straightforward starting approach is a GRU.

Inputs:

\[
z_t=E(o_t)
\]

Recurrent update:

\[
h_t=\operatorname{GRU}(z_t,h_{t-1})
\]

Predicted state:

\[
\hat x_t=D(h_t)
\]

The tactile encoder converts fingertip arrays into compact features.

These features are concatenated with proprioceptive information.

The recurrent network maintains temporal context.

Outputs may include:

- Estimated relative layer angle.
- Estimated cube orientation.
- Contact-state classification.
- Slip probability.
- Confidence in completion.

### Training data

Simulation makes supervised estimator training relatively straightforward.

Collect trajectories containing:

- Tactile measurements.
- Proprioceptive measurements.
- Executed actions.
- Ground-truth cube pose.
- Ground-truth layer angles.
- Ground-truth contact information.

Ground-truth values are training labels, not permitted deployment inputs.

### Initial training objective

Use a combination of:

\[
\mathcal L=
\lambda_p\mathcal L_{\text{pose}}
+\lambda_\theta\mathcal L_{\text{angle}}
+\lambda_c\mathcal L_{\text{contact}}
+\lambda_s\mathcal L_{\text{slip}}
\]

The loss functions should account for the geometry of the represented variables.

For example, angular losses should use wrapped angular differences rather than ordinary unbounded subtraction.

---

## 16. Advanced estimator: belief-state tracking

A point estimate may be insufficient when tactile measurements are ambiguous.

Suppose a turn is attempted.

The robot is uncertain whether:

- The turn succeeded.
- The turn stopped halfway.
- The entire cube moved.
- A slip prevented meaningful rotation.

Instead of committing immediately to one interpretation, maintain several hypotheses.

A particle-based estimator can represent:

\[
b_t(x)\approx
\sum_{i=1}^{N}
w_t^{(i)}\delta(x-x_t^{(i)})
\]

Each particle represents a possible physical state.

The particles are propagated using action-conditioned dynamics and reweighted using sensor observations.

This approach is motivated by the 2023 modular tactile-manipulation work.

### Hybrid symbolic and continuous belief

A useful extension is to maintain both:

1. A continuous physical-state belief.
2. A discrete distribution over possible cube configurations.

For example:

| Hypothesis | Probability |
|---|---:|
| Intended turn completed | 0.82 |
| Intended turn did not occur | 0.10 |
| Wrong layer turned | 0.05 |
| Other uncertain outcome | 0.03 |

These values are illustrative.

The probabilities should be learned or calibrated from real simulation data.

This hybrid representation prevents the system from silently corrupting its symbolic cube state after an uncertain manipulation.

---

# Part VII — World Model

## 17. Purpose of the world model

The project does not strictly require a large learned world model.

A recurrent estimator plus symbolic planner may already be sufficient for simplified blindfolded solving.

Therefore, the world model should be introduced as a research extension and compared against the simpler system.

Its purpose is to predict the consequences of candidate manipulation actions.

Potential predictions include:

- Future tactile measurements.
- Future joint configurations.
- Future layer angles.
- Contact changes.
- Slip events.
- Probability of a successful turn.

---

## 18. World-model architecture

Define an observation encoder:

\[
z_t=E_\phi(o_t)
\]

An action-conditioned transition model predicts the next latent state:

\[
\hat z_{t+1}=F_\theta(z_t,a_t)
\]

A decoder predicts relevant physical quantities:

\[
\hat y_{t+1}=D_\psi(\hat z_{t+1})
\]

where:

\[
y_t=
(T_t,\theta_t,R_t,c_t)
\]

For richer temporal modeling, use a recurrent latent state:

\[
h_{t+1}=F_\theta(h_t,z_t,a_t)
\]

The model should support multi-step prediction, not only one-step reconstruction.

### Prediction horizons

A useful starting design is:

- 1 control step.
- 5 control steps.
- 10 control steps.

This is motivated by FINGR's multi-horizon future-interaction supervision.

Short horizons capture immediate contact changes.

Longer horizons capture progress through a turn and subsequent finger recovery.

### World-model training loss

A possible objective is:

\[
\mathcal L_{\text{WM}}
=
\lambda_T\mathcal L_{\text{tactile}}
+
\lambda_\theta\mathcal L_{\text{angle}}
+
\lambda_q\mathcal L_{\text{joints}}
+
\lambda_c\mathcal L_{\text{contact}}
\]

Additional uncertainty-aware losses should be considered if the model predicts distributions.

---

## 19. Using the world model for control

There are two main approaches.

### Approach A — Predictive features for the RL policy

The world model produces features describing expected future contact and motion.

These features become additional inputs to the manipulation policy.

Advantages:

- Relatively straightforward integration.
- Low inference overhead.
- Similar in spirit to FINGR's predictive feature learning.

### Approach B — Model-predictive control

The world model evaluates candidate action sequences.

The controller selects actions expected to produce stable progress.

Conceptually:

\[
a_{t:t+H}^{*}
=
\arg\max_{a_{t:t+H}}
\mathbb E
\left[
\sum_{k=t}^{t+H}
r(\hat x_k,a_k)
\right]
\]

Only the first action or a short initial segment is executed.

The robot then observes new tactile information and replans.

Advantages:

- Explicit use of predicted consequences.
- Natural integration of uncertainty.
- Potentially better recovery from unexpected contacts.

Disadvantages:

- Increased implementation complexity.
- Model prediction errors.
- Additional computation.
- Difficulty searching high-dimensional joint-action spaces.

The recommended sequence is to begin with predictive features and introduce model-predictive control only after the learned dynamics demonstrate useful accuracy.

---

# Part VIII — Blindfolded Cube Solving

## 20. Visual inspection

The robot must reconstruct the cube configuration before visual feedback is removed.

For the first version, this subsystem can be simplified.

### Baseline inspection method

Initialize the cube in a known scrambled state.

Provide the initial symbolic configuration to the planner.

This is equivalent to assuming perfect visual inspection.

It is a useful control experiment, but it should not be described as a complete visual-inspection system.

### Full inspection method

The robot uses rendered camera images to inspect the cube.

It must:

1. Identify visible cube faces.
2. Determine sticker colors.
3. Associate observed stickers with cube coordinates.
4. Move or rotate the cube to reveal hidden faces.
5. Reconstruct a legal symbolic cube state.
6. Validate consistency.
7. Commit the state to memory.

Only then is vision disabled.

The inspection system may be developed independently of the tactile manipulation system.

### Important distinction

Perfectly remembering the initial puzzle configuration does not imply knowing the cube's future physical state.

The symbolic configuration remains known only if subsequent physical actions are correctly estimated.

---

## 21. Symbolic cube solver

Use an existing solving algorithm.

For the 2×2×2 cube, a compact state-space search or lookup-table solver is suitable.

The solver produces a sequence:

\[
\Pi=[m_1,m_2,\ldots,m_K]
\]

where each move is drawn from the supported cube-turn notation.

The planner should support:

- Clockwise turns.
- Counterclockwise turns.
- Double turns where appropriate.
- Whole-cube reorientation.
- Regrasping operations.

### Planning for physical execution

The shortest symbolic solution is not necessarily the easiest physical solution.

For example, a solution containing fewer moves may require difficult regrasping transitions.

A more physically informed objective is:

\[
J(\Pi)=
\alpha N_{\text{moves}}
+\beta N_{\text{regrasps}}
+\gamma C_{\text{difficulty}}
+\delta C_{\text{risk}}
\]

where:

- \(N_{\text{moves}}\): number of symbolic moves.
- \(N_{\text{regrasps}}\): expected regrasp operations.
- \(C_{\text{difficulty}}\): estimated manipulation difficulty.
- \(C_{\text{risk}}\): estimated failure risk.

This provides an opportunity to integrate learned physical manipulation statistics into symbolic planning.

A risk-aware planner may choose a longer but more reliable sequence.

---

## 22. Execution verification

After each commanded face turn, the system must decide whether the move was completed.

For example:

\[
m_t=U
\]

The controller attempts a 90° upper-layer rotation.

The state estimator predicts:

\[
\hat\theta_t=89.2^\circ
\]

with an associated uncertainty.

The execution monitor then determines whether the move can be considered complete.

The exact acceptance threshold should be determined experimentally.

Potential criteria include:

- Estimated angle near the target.
- Low estimated angular velocity.
- Stable contact configuration.
- Low probability of unintended layer movement.
- Sufficient confidence in the state estimate.

A move should not be recorded as successful solely because the controller issued the correct command.

---

## 23. Recovery strategies

Recovery is one of the most important parts of the project.

### Case A — Incomplete rotation

The estimator detects that the intended layer has not reached its target.

Action:

Apply a corrective motion while maintaining support.

### Case B — No effective rotation

The finger moved but the layer did not.

Action:

Reposition the manipulating finger and retry.

### Case C — Whole-cube movement

The cube rotates relative to the hand.

Action:

Stabilize the grasp and update the estimated whole-cube pose.

### Case D — Wrong-layer rotation

The estimator detects evidence that an unintended layer moved.

Action:

Update the symbolic-state belief, recover if possible, and replan.

### Case E — Ambiguous state

The robot cannot confidently determine what occurred.

Action:

Avoid blindly advancing to the next symbolic move.

Instead, perform a safe tactile verification action or recover a known configuration.

### Case F — Cube dropped

The robot loses control of the cube.

For the initial benchmark, terminate the episode.

An advanced version may attempt retrieval and regrasping, but this introduces a substantially different problem.

---

# Part IX — Training Strategy

## 24. Recommended curriculum

### Stage 0 — Pure simulation validation

No learning.

Validate:

- Robot model.
- Actuators.
- Collisions.
- Cube mechanics.
- Tactile signals.
- Episode reset logic.

**Deliverable:** Stable environment with reproducible physics.

### Stage 1 — Grasp stabilization

Train the hand to maintain a stable grasp.

Initially allow privileged state information for reward calculation and, if necessary, teacher-policy training.

**Deliverable:** Reliable sustained grasp without vision.

### Stage 2 — Rigid cube rotation

Train whole-cube reorientation.

This is a useful stepping stone toward articulated manipulation.

**Deliverable:** Rotate the cube between selected orientations while maintaining grasp stability.

### Stage 3 — Single-layer rotation

Introduce one rotatable cube face.

**Deliverable:** Execute a commanded 90° face turn.

### Stage 4 — Tactile estimation

Collect manipulation trajectories and train a model to estimate physical layer motion.

**Deliverable:** Estimate turn progress from tactile/proprioceptive observations.

### Stage 5 — Blindfolded single turns

Remove all privileged cube-state inputs from the deployed controller.

**Deliverable:** Execute and verify a face turn without vision.

### Stage 6 — Multi-turn sequences

Execute sequences of two, three, five, and progressively more moves.

**Deliverable:** Maintain state consistency over repeated manipulation.

### Stage 7 — Restricted cube solving

Solve cubes scrambled by short legal move sequences.

**Deliverable:** Complete blindfolded solutions of limited scrambles.

### Stage 8 — General 2×2×2 solving

Increase scramble diversity and sequence length.

**Deliverable:** Solve randomly scrambled cubes under the strict blindfolded evaluation protocol.

### Stage 9 — Learned world-model integration

Compare the baseline controller against versions with predictive state modeling.

**Deliverable:** Quantitative evaluation of whether world-model predictions improve manipulation reliability.

---

## 25. Training with privileged information

Privileged simulator state can be useful during training.

For example, an asymmetric actor–critic architecture may provide:

**Actor:**

\[
o_t^{actor}
=
(q_t,\dot q_t,T_t,\text{goal},\text{history})
\]

**Critic:**

\[
o_t^{critic}
=
(o_t^{actor},x_t^{true})
\]

The critic receives additional simulation information that can improve value estimation.

The actor does not.

Another option is teacher–student learning.

A teacher policy receives accurate cube pose and layer-angle information.

A student policy learns to reproduce the teacher's actions using only tactile and proprioceptive observations.

This strategy may substantially simplify the initial learning problem.

However, the final evaluation must use the student policy without access to the teacher's privileged observations.

---

## 26. Domain randomization

Domain randomization should be introduced gradually.

Relevant parameters include:

- Cube mass.
- Center of mass.
- Contact friction.
- Contact stiffness.
- Joint damping.
- Actuator response.
- Tactile measurement noise.
- Sensor latency.
- Initial cube pose.
- Initial finger configuration.
- Small collision-geometry variations.

### Recommended procedure

Begin with a deterministic simulator.

Establish that the policy can reliably perform the target manipulation.

Then introduce progressively larger randomization.

Avoid changing many unrelated parameters simultaneously during early development.

This makes it easier to identify why performance changes.

Although the project does not initially target sim-to-real transfer, randomization remains useful for testing whether the controller learned robust manipulation behavior rather than exploiting a narrow simulator configuration.

---

# Part X — Experimental Design

## 27. Primary research questions

### RQ1 — Does tactile sensing improve blindfolded layer-turn execution?

Compare:

- Proprioception only.
- Proprioception plus scalar contact signals.
- Proprioception plus fingertip force observations.
- Proprioception plus distributed tactile taxels.

### RQ2 — Does explicit state estimation improve long-horizon manipulation?

Compare:

- Reactive controller.
- Recurrent controller.
- Controller with explicit state estimator.
- Controller with uncertainty-aware belief estimation.

### RQ3 — Does a learned world model improve execution reliability?

Compare:

- Model-free RL.
- Model-free RL with predictive auxiliary features.
- World-model-assisted control.

### RQ4 — Can the robot recover from unexpected manipulation errors?

Evaluate robustness to:

- Partial turns.
- Slip events.
- Whole-cube disturbances.
- Unexpected contact loss.
- Changes in friction.

### RQ5 — How does performance scale with sequence length?

Measure success for increasingly long symbolic move sequences.

This is critical because even small per-turn failure rates accumulate over long sequences.

---

## 28. Evaluation metrics

### 28.1 Individual turn success rate

\[
SR_{\text{turn}}=
\frac{N_{\text{successful turns}}}
{N_{\text{attempted turns}}}
\]

### 28.2 Sequence success rate

\[
SR_{\text{sequence}}=
\frac{N_{\text{successful sequences}}}
{N_{\text{attempted sequences}}}
\]

### 28.3 Complete solve success rate

\[
SR_{\text{solve}}=
\frac{N_{\text{solved cubes}}}
{N_{\text{scrambled cubes}}}
\]

### 28.4 State-estimation error

For layer angle:

\[
e_\theta=
|\operatorname{wrap}(\hat\theta-\theta)|
\]

Use appropriate geometric error measures for cube orientation.

### 28.5 Error detection performance

Measure:

- Precision.
- Recall.
- False-positive rate.
- False-negative rate.
- Detection delay.

### 28.6 Recovery performance

\[
SR_{\text{recovery}}=
\frac{N_{\text{recovered failures}}}
{N_{\text{recoverable failures}}}
\]

### 28.7 Training efficiency

Report:

- Environment interactions.
- Simulator wall-clock time.
- Total training time.
- Hardware configuration.
- Number of parallel environments.

These quantities make experiments reproducible and allow fair comparisons between learning methods.

---

## 29. Evaluation protocol

A recommended benchmark should contain several groups.

| Benchmark | Description |
|---|---|
| B1 | Single-layer 90° turn |
| B2 | Clockwise/counterclockwise turn |
| B3 | Two consecutive turns |
| B4 | Five consecutive turns |
| B5 | Short-scramble 2×2×2 solving |
| B6 | Random-scramble 2×2×2 solving |
| B7 | Disturbed execution |
| B8 | Generalization to randomized physical parameters |

For each benchmark:

- Use independent training and evaluation seeds.
- Keep test configurations separate from training configurations.
- Report multiple training runs.
- Include confidence intervals.
- Record both success and failure trajectories.
- Preserve simulator and code versions.
- Evaluate under the same physical randomization for all compared methods.

A practical initial target is at least 100 evaluation episodes per major condition, with larger samples for final success-rate claims.

For long-horizon solving, record not only whether the puzzle was solved but also the first point at which execution diverged from the intended symbolic sequence.

---

# Part XI — Software Architecture

## 30. Proposed repository structure

```text
blindfolded-cube/
│
├── assets/
│   ├── robots/
│   │   ├── sharpa_wave/
│   │   └── franka/
│   └── cubes/
│       ├── single_axis/
│       ├── restricted_cube/
│       └── cube_2x2/
│
├── simulation/
│   ├── scene_builder.py
│   ├── robot_interface.py
│   ├── cube_interface.py
│   ├── contact_model.py
│   └── tactile_sensors.py
│
├── environments/
│   ├── grasp_env.py
│   ├── rigid_rotation_env.py
│   ├── face_turn_env.py
│   ├── sequence_env.py
│   └── blindfolded_solve_env.py
│
├── perception/
│   ├── visual_inspection.py
│   ├── tactile_encoder.py
│   ├── state_estimator.py
│   └── belief_filter.py
│
├── policies/
│   ├── grasp_policy.py
│   ├── turn_policy.py
│   ├── teacher_policy.py
│   └── student_policy.py
│
├── world_models/
│   ├── encoder.py
│   ├── dynamics.py
│   ├── decoder.py
│   └── trainer.py
│
├── planning/
│   ├── cube_state.py
│   ├── cube_solver.py
│   ├── skill_planner.py
│   └── recovery_planner.py
│
├── training/
│   ├── train_grasp.py
│   ├── train_turn.py
│   ├── train_estimator.py
│   └── train_world_model.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── ablations.py
│   └── visualization.py
│
├── configs/
│   ├── robot.yaml
│   ├── tactile.yaml
│   ├── rl.yaml
│   └── experiments.yaml
│
├── tests/
│   ├── test_cube_kinematics.py
│   ├── test_tactile.py
│   ├── test_observation_leakage.py
│   └── test_symbolic_transitions.py
│
└── README.md
```

---

## 31. Critical software interfaces

The project should define stable interfaces early.

### Robot interface

Responsibilities:

- Read joint states.
- Apply actuator commands.
- Reset robot configuration.
- Report actuator limits.

### Tactile interface

Responsibilities:

- Read sensor observations.
- Apply sensor preprocessing.
- Introduce configurable noise.
- Return a consistent tensor layout.

### Cube interface

Responsibilities:

- Reset cube configuration.
- Apply legal symbolic scrambles.
- Track the ground-truth puzzle state for evaluation.
- Expose physical-state measurements only through authorized interfaces.

### Policy interface

Inputs:

- Proprioception.
- Tactile observations.
- Manipulation goal.
- Optional recurrent memory.

Outputs:

- Joint commands.
- Optional predicted skill status.

### State-estimator interface

Inputs:

- Observation history.
- Action history.

Outputs:

- Estimated physical state.
- Confidence or uncertainty.
- Predicted contact and turn status.

### Planner interface

Inputs:

- Symbolic cube-state belief.
- Available manipulation skills.
- Estimated physical state.

Outputs:

- Next manipulation goal.
- Recovery instruction when necessary.

Defining these interfaces early will make it easier to compare alternative learning architectures without rewriting the entire simulator.

---

# Part XII — Practical Development Plan

## 32. Timeline

The following timeline assumes a technically experienced developer working part-time.

It is a planning estimate, not a guarantee.

| Phase | Activities | Estimated duration |
|---|---|---|
| 1 | Research and environment setup | 1–2 weeks |
| 2 | Cube physics and robot integration | 2–4 weeks |
| 3 | Tactile sensing validation | 1–3 weeks |
| 4 | Grasp and rigid-object manipulation | 2–4 weeks |
| 5 | Single-layer turning policy | 3–6 weeks |
| 6 | Tactile state estimation | 2–4 weeks |
| 7 | Blindfolded multi-turn execution | 3–6 weeks |
| 8 | Restricted cube solving | 2–5 weeks |
| 9 | World-model experiments | 3–6 weeks |
| 10 | Evaluation and documentation | 2–3 weeks |

Some phases can proceed in parallel.

### Overall estimate

**A strong intermediate research prototype may require approximately 3–5 months of part-time development.**

A complete blindfolded 2×2×2 solving demonstration may require 6–12 months or more, depending on manipulation reliability and the amount of reusable infrastructure available.

These estimates exclude a complete redesign of the cube physics model if the initial representation proves unsuitable.

---

## 33. Development milestones and acceptance gates

### Milestone M0 — Simulation foundation

**Goal:** Reliable cube and robot simulation.

Acceptance requirements:

- Robot loads successfully.
- Actuators behave correctly.
- Cube rotations preserve legal geometry.
- No unacceptable instability during repeated interactions.

### Milestone M1 — Tactile foundation

**Goal:** Reliable simulated tactile sensing.

Acceptance requirements:

- Detect contacts.
- Detect changes in contact location.
- Capture useful sliding information.
- Produce consistent observations.

### Milestone M2 — Single-turn policy

**Goal:** Execute a commanded layer turn.

Suggested target:

At least 90% success under the fixed nominal simulation configuration.

This is a proposed engineering threshold, not a result established by the literature.

### Milestone M3 — Blindfolded estimator

**Goal:** Estimate layer progress without privileged observations.

Suggested target:

Mean absolute angular error below 10° during controlled turn trajectories, with sufficiently calibrated uncertainty to detect ambiguous cases.

The final acceptable error should be determined by the mechanical tolerances required for successful turns.

### Milestone M4 — Multi-turn execution

**Goal:** Execute five consecutive commanded turns.

Suggested target:

At least 80% complete-sequence success under nominal conditions.

### Milestone M5 — Restricted solving

**Goal:** Solve cubes scrambled by short sequences.

Suggested target:

Reliable completion of three-move scrambles before increasing complexity.

### Milestone M6 — World-model comparison

**Goal:** Establish whether learned predictive modeling improves performance.

Acceptance requirements:

- Reproducible baselines.
- Comparable training budgets.
- Multiple evaluation seeds.
- Measurable improvement on at least one predefined metric.

A negative result is still scientifically useful if the evaluation is rigorous.

---

# Part XIII — Risk Analysis

## 34. Major technical risks

### Risk 1 — Cube physics is unstable

**Severity:** High

The cube's articulated structure may be difficult to simulate faithfully.

Mitigation:

- Begin with one rotatable layer.
- Validate long sequences of externally driven legal moves.
- Use simplified collision geometry.
- Separate symbolic state from physical dynamics.

### Risk 2 — Tactile sensing is insufficient

**Severity:** High

Fingertip-only observations may not provide enough information to determine the cube's state.

Mitigation:

- Compare multiple tactile configurations.
- Add palm or finger-link contact sensing if needed.
- Maintain temporal observation history.
- Use uncertainty-aware estimation.
- Explicitly test which physical motions are observable.

### Risk 3 — RL fails to discover stable turns

**Severity:** High

The action space is large and contact dynamics are discontinuous.

Mitigation:

- Reduce active degrees of freedom.
- Use demonstrations.
- Train individual skills.
- Introduce curriculum learning.
- Begin with fixed wrist and stable initial grasps.

### Risk 4 — GPU acceleration is incompatible with sensor features

**Severity:** Medium–high

Not every MuJoCo feature is supported identically across CPU and accelerated backends.

Mitigation:

- Validate standard MuJoCo first.
- Test sensor compatibility independently.
- Benchmark throughput.
- Maintain a CPU reference implementation.

### Risk 5 — State estimation becomes unreliable

**Severity:** High

Small errors may accumulate over repeated turns.

Mitigation:

- Estimate uncertainty.
- Validate state-estimator calibration.
- Detect ambiguous transitions.
- Add verification and recovery actions.
- Avoid advancing the symbolic plan after uncertain turns.

### Risk 6 — End-to-end solving is too difficult

**Severity:** High

The project may achieve individual face turns without reliably solving full scrambles.

Mitigation:

- Define intermediate milestones as standalone deliverables.
- Prioritize reliable turn execution and state estimation.
- Use symbolic planning instead of learning puzzle-solving logic from scratch.
- Treat arbitrary-scramble solving as an advanced objective.

---

# Part XIV — Recommended Minimum Viable Project

## 35. Initial project scope

The first complete experiment should be narrower than full blindfolded cube solving.

### Proposed MVP

**Train a dexterous hand to rotate one articulated cube layer by 90° and verify the rotation using only tactile and proprioceptive observations.**

The environment should provide:

- A fixed or controlled wrist.
- A stable initial grasp.
- A cube with one active layer.
- Fingertip tactile sensing.
- A target rotation command.

The policy should not receive the true layer angle.

A learned estimator should predict the angle from tactile/proprioceptive history.

The manipulation controller should use that estimate to determine when to stop.

### Why this is the correct starting point

This task isolates the central research challenge:

**Can a robot measure the result of its own articulated-object manipulation through touch?**

If the answer is yes, longer sequences become an integration and robustness problem.

If the answer is no, adding a full Rubik's Cube solver will not solve the underlying sensing limitation.

---

## 36. First three experiments

### Experiment 1 — Observability

Collect sensor data from known layer rotations.

Train a model to estimate the layer angle.

Compare:

- Proprioception only.
- Fingertip force sensing.
- Distributed tactile taxels.

**Research question:** Can the selected sensors distinguish completed and incomplete turns?

### Experiment 2 — Closed-loop manipulation

Train a controller to rotate the layer toward a requested angle.

Compare:

- Controller using ground-truth layer angle.
- Controller using estimated layer angle.
- Controller without explicit layer-angle feedback.

**Research question:** How much performance is lost when privileged state is replaced with tactile estimation?

### Experiment 3 — Disturbance recovery

Introduce controlled execution errors.

Examples:

- Temporarily reduce friction.
- Interrupt a turn.
- Perturb the cube orientation.

Measure whether the estimator detects the error and whether the controller can recover.

**Research question:** Can tactile feedback correct manipulation errors rather than merely support nominal execution?

These three experiments already constitute a meaningful project.

---

# Part XV — Recommended Research Priorities

## 37. Priority order

The project contains several interesting machine-learning problems.

They should not all be pursued simultaneously.

### Priority 1 — Manipulation feasibility

Determine whether the simulated hand can reliably perform the necessary physical actions.

### Priority 2 — Tactile observability

Determine whether the sensor interface provides enough information to estimate the relevant physical state.

### Priority 3 — Closed-loop control

Use tactile state estimates to improve physical execution.

### Priority 4 — Long-horizon memory

Track the cube's symbolic and physical state across repeated actions.

### Priority 5 — Recovery

Detect failed actions and restore a consistent state.

### Priority 6 — World models

Investigate whether action-conditioned predictive modeling improves robustness or sample efficiency.

### Priority 7 — Visual inspection

Replace the initial-state oracle with an actual camera-based inspection system.

### Priority 8 — VLA integration

Introduce language-conditioned planning or manipulation goals.

The VLA component is intentionally last.

A language-conditioned interface is not necessary to demonstrate the central tactile state-estimation contribution.

---

# Part XVI — Further Research Directions

## 38. Active tactile perception

Instead of passively observing contacts, the robot could perform deliberate actions to reduce uncertainty.

For example:

- Apply a small probing torque.
- Change finger pressure.
- Shift contact location.
- Test whether a layer is aligned.
- Reposition the cube into a better grasp.

The objective is not immediately to advance the puzzle solution but to acquire information.

An information-gathering action can be selected using:

\[
a_t^{*}
=
\arg\max_a
\left[
H(b_t)
-
\mathbb E[H(b_{t+1})\mid a]
\right]
\]

where \(H\) is entropy.

This extends the project toward active perception and belief-space planning.

---

## 39. Risk-aware symbolic planning

A conventional Rubik's Cube solver minimizes symbolic move count.

A physical robot should consider manipulation success probability.

If one type of turn is particularly unreliable, it may be preferable to choose a longer solution composed of easier turns.

For candidate plans:

\[
P(\text{success}\mid\Pi)
\]

can be estimated from empirical manipulation statistics.

A risk-aware solver could optimize both the expected execution cost and the probability of complete success.

This is a promising interaction between symbolic reasoning and learned physical skills.

---

## 40. Generalization beyond Rubik's Cubes

The same architecture could be adapted to:

- Blindfolded mechanical assembly.
- Rotating knobs or dials under occlusion.
- Manipulating articulated tools.
- Connector insertion without visual feedback.
- Operating mechanical combination locks.
- Aligning components inside enclosed spaces.

These applications share the need to estimate hidden mechanical state from contact and action history.

Therefore, a successful project could contribute a reusable methodology for tactile manipulation under partial observability.

---

# Part XVII — Final Recommendations

## 41. Recommended implementation decisions

| Decision | Recommendation |
|---|---|
| Initial cube | 2×2×2 |
| Initial hand | Sharpa Wave |
| Initial arm | Fixed wrist |
| Full arm integration | Introduce after single-turn success |
| Physics engine | Standard MuJoCo |
| Sensor baseline | Simple contact/force observations |
| Rich sensing | Distributed fingertip taxels |
| First learning algorithm | SAC |
| Initial controller | Joint-position targets |
| State estimator | GRU |
| Advanced estimator | Differentiable or conventional particle filter |
| Initial symbolic planner | Existing 2×2×2 solver |
| Initial world model | Action-conditioned latent dynamics |
| Advanced control | Belief-aware planning or MPC |
| Initial inspection | Ground-truth initial-state oracle |
| Final inspection | Camera-based cube-state reconstruction |
| Main evaluation | Blindfolded physical execution reliability |

---

## 42. Recommended order of execution

The first development cycle should focus entirely on simulation mechanics.

The second should establish meaningful tactile observations.

The third should train a single-layer turning policy.

The fourth should replace privileged angle feedback with tactile state estimation.

The fifth should introduce sequences and symbolic memory.

Only after these components work reliably should the project attempt complete blindfolded solving.

This progression reduces the chance of spending substantial time debugging a system in which physics, sensing, control, state estimation, and planning are all failing simultaneously.

---

## 43. Final project definition

**Working title:**

*BlindCube: Tactile World Models for Blindfolded Dexterous Rubik's Cube Manipulation*

**Research objective:**

Develop and evaluate a hierarchical robotic system that visually inspects an articulated cube, retains its symbolic configuration, and executes manipulation sequences using tactile and proprioceptive feedback without further visual observations.

**Core hypothesis:**

Explicit tactile state estimation and action-conditioned predictive modeling can improve the reliability of long-horizon articulated-object manipulation when visual feedback is unavailable.

**Minimum deliverable:**

A tactile-controlled robotic hand capable of executing and verifying individual cube-layer turns without visual or privileged object-state feedback.

**Intermediate deliverable:**

A robotic hand capable of executing short sequences of cube turns while maintaining a consistent internal state estimate.

**Target deliverable:**

A robotic system capable of solving restricted scrambles of a 2×2×2 Rubik's Cube after an initial visual inspection.

**Stretch deliverable:**

Reliable blindfolded solving of arbitrary 2×2×2 scrambles, with explicit uncertainty tracking, error detection, and recovery.

**Long-term extension:**

Generalization to 3×3×3 cubes and other articulated mechanical objects.

---

# Appendix A — Essential Resources

### Simulation and robot models

**MuJoCo**
https://mujoco.org/

**MuJoCo Menagerie**
https://github.com/google-deepmind/mujoco_menagerie

**Sharpa Wave model**
https://github.com/google-deepmind/mujoco_menagerie/tree/main/sharpa_wave

**MuJoCo tactile sensor reference**
https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-tactile

**MuJoCo MJX**
https://mujoco.readthedocs.io/en/stable/mjx.html

**OpenAI RoboGym**
https://github.com/openai/robogym

### Core research papers

**OpenAI Dactyl — Rubik's Cube**
https://openai.com/index/solving-rubiks-cube/

**Learning Purely Tactile In-Hand Manipulation**
https://arxiv.org/abs/2204.03698

**Dextrous Tactile In-Hand Manipulation Using a Modular Reinforcement Learning Architecture**
https://arxiv.org/abs/2303.04705

**Rotating without Seeing**
https://arxiv.org/abs/2303.10880

**FINGR**
https://arxiv.org/abs/2609.33973

**DexTouch-WM**
https://arxiv.org/abs/2609.20649

### Additional research directions

**AnyRotate**
https://arxiv.org/abs/2405.07391

**DexTouch: Learning to Seek and Manipulate Objects with Tactile Dexterity**
https://arxiv.org/abs/2401.12496

**Sim2Real Manipulation on Unknown Objects with Tactile-Based Reinforcement Learning**
https://arxiv.org/abs/2403.12170

---

# Appendix B — Immediate Action Plan

The first implementation sprint should aim to answer a small number of concrete questions.

**Task 1: Robot integration**

Load the Sharpa Wave model in MuJoCo and verify actuator control.

Deliverable: a working simulated hand with reproducible joint commands.

**Task 2: Sensor integration**

Implement a tactile sensor on one fingertip and validate its response to a cube pressing and sliding against it.

Deliverable: recorded tactile observations from controlled contact experiments.

**Task 3: Cube mechanics**

Implement a cube with one physically rotatable layer.

Deliverable: a simulation that can repeatedly execute legal 90° layer turns without instability.

**Task 4: Manual manipulation baseline**

Develop a scripted or manually controlled sequence that establishes contact and attempts a layer rotation.

Deliverable: at least one reproducible physical face turn.

**Task 5: Initial dataset**

Collect trajectories containing tactile observations, proprioception, actions, and ground-truth layer angles.

Deliverable: a dataset suitable for training a layer-angle estimator.

**Task 6: First machine-learning experiment**

Train a recurrent model to predict layer angle from tactile/proprioceptive history.

Deliverable: a quantitative comparison against a proprioception-only baseline.

Once these six tasks are complete, the project will have the simulator, sensors, dataset, and first estimator required to begin learning closed-loop blindfolded manipulation.

---

# Conclusion

Blindfolded robotic Rubik's Cube solving is a challenging but well-motivated research direction.

Its difficulty arises primarily from the interaction between contact-rich physical manipulation and partial observability.

The symbolic puzzle-solving component is comparatively straightforward once the initial cube configuration is known.

The critical problems are:

- Achieving reliable physical layer turns.
- Estimating hidden mechanical state from tactile observations.
- Maintaining a consistent internal representation over long action sequences.
- Detecting and correcting manipulation errors.
- Managing uncertainty when physical outcomes cannot be observed directly.

Recent work on tactile in-hand manipulation, modular state estimation, and robotic Rubik's Cube solving provides a strong foundation.

The recommended project should use that foundation rather than attempting to rediscover every component from scratch.

The most valuable early research result would be a demonstration that a robot can **execute, measure, and correct a cube-layer rotation using touch alone**.

That capability is the foundation on which reliable blindfolded Rubik's Cube solving—and more general tactile world-model-based manipulation—can be built.
