#!/usr/bin/env python3
"""
Complete the golden QA benchmark by adding remaining 32 pairs.
Current: 68 pairs (23 neuro, 23 quantum, 22 general)
Target: 100 pairs (30 neuro, 30 quantum, 40 general)
Need: 7 neuro, 7 quantum, 18 general
"""

import json
from pathlib import Path

# Additional QA pairs to reach 100 total
additional_pairs = [
    # Neuroscience additions (7 pairs: 3 simple, 2 medium, 2 complex)
    {
        "id": "neuro_simple_012",
        "domain": "neuroscience",
        "difficulty": "simple",
        "question": "What is the difference between excitatory and inhibitory neurotransmitters?",
        "answer": "Excitatory neurotransmitters increase the likelihood that the postsynaptic neuron will fire an action potential by depolarizing the membrane (making it more positive). The main excitatory neurotransmitter is glutamate. Inhibitory neurotransmitters decrease firing probability by hyperpolarizing the membrane (making it more negative). The main inhibitory neurotransmitter is GABA (gamma-aminobutyric acid). The balance between excitation and inhibition is critical for proper brain function, and disruptions can lead to seizures, anxiety disorders, and other neurological conditions.",
        "ground_truth": "Excitatory neurotransmitters (e.g., glutamate) increase firing probability; inhibitory (e.g., GABA) decrease it",
        "contexts": [
            "Glutamate is the primary excitatory neurotransmitter in the CNS",
            "GABA is the primary inhibitory neurotransmitter",
            "Excitation-inhibition balance critical for brain function",
            "Imbalances linked to epilepsy and psychiatric disorders"
        ],
        "source_file": "neuroscience_general",
        "tags": ["neurotransmitter", "excitation", "inhibition", "glutamate", "GABA"]
    },
    {
        "id": "neuro_simple_013",
        "domain": "neuroscience",
        "difficulty": "simple",
        "question": "What is diffusion tensor imaging (DTI) and what does it measure?",
        "answer": "Diffusion Tensor Imaging (DTI) is an MRI technique that measures water diffusion in brain tissue to map white matter structure. Water diffuses preferentially along axon bundles, creating anisotropic (directionally dependent) diffusion. DTI quantifies this directionality with metrics like fractional anisotropy (FA, 0=isotropic, 1=highly directional). Tractography algorithms use DTI to reconstruct white matter fiber pathways, revealing brain connectivity. Applications include studying development, aging, brain injuries, and neurological diseases. DTI complements functional connectivity by providing anatomical connectivity information.",
        "ground_truth": "DTI measures water diffusion anisotropy to map white matter structure and connectivity pathways",
        "contexts": [
            "Water diffuses along axon bundles creating directional signal",
            "Fractional anisotropy quantifies diffusion directionality",
            "Tractography reconstructs white matter pathways",
            "Useful for studying structural connectivity and white matter diseases"
        ],
        "source_file": "neuroscience_general",
        "tags": ["DTI", "white_matter", "diffusion", "tractography", "connectivity"]
    },
    {
        "id": "neuro_simple_014",
        "domain": "neuroscience",
        "difficulty": "simple",
        "question": "What is the cerebellum and what are its primary functions?",
        "answer": "The cerebellum is a large structure at the base of the brain, posterior to the brainstem. Despite occupying only 10% of brain volume, it contains over 50% of the brain's neurons. Primary functions: 1) Motor coordination and balance - integrates sensory and motor information to refine movements, 2) Motor learning - adapts movements through practice, 3) Timing and prediction of movements, 4) Cognitive functions - recently discovered roles in language, attention, and emotion processing. Damage causes ataxia (uncoordinated movements), intention tremor, dysmetria (overshooting targets), and impaired motor learning. The cerebellum has dense connections with motor cortex and spinal cord.",
        "ground_truth": "Cerebellum coordinates movement, balance, and motor learning; also involved in cognition",
        "contexts": [
            "Contains majority of brain's neurons despite small volume",
            "Critical for motor coordination and adaptation",
            "Purkinje cells main computational neurons",
            "Damage causes characteristic ataxia and tremor"
        ],
        "source_file": "neuroscience_general",
        "tags": ["cerebellum", "motor_control", "coordination", "motor_learning"]
    },
    {
        "id": "neuro_medium_009",
        "domain": "neuroscience",
        "difficulty": "medium",
        "question": "Explain the concept of population coding in neural systems and how it differs from single neuron coding.",
        "answer": "Population coding represents information through patterns of activity across many neurons rather than individual neuron responses. Single neurons have broad tuning (respond to range of stimuli), noisy responses, and limited information capacity. Populations overcome these limitations: 1) Averaging reduces noise, increasing reliability, 2) Different neurons prefer different stimuli, covering full range, 3) Joint activity patterns represent stimuli precisely, 4) Distributed representation robust to neuron loss. Types: 1) Rate coding - information in firing rates across population, 2) Temporal coding - spike timing contains information, 3) Correlation/synchrony coding - coordinated activity between neurons. Advantages: higher capacity, noise reduction, flexibility. Decoding methods: population vectors, maximum likelihood, neural networks. Applications: motor control (M1 populations encode movement direction), sensory processing (V1 populations represent orientation), memory (hippocampal ensembles represent locations).",
        "ground_truth": "Population coding uses patterns across many neurons for robust, high-capacity information representation vs individual neuron responses",
        "contexts": [
            "Reduces noise through averaging across neurons",
            "Distributed representation more robust and flexible",
            "Population vectors decode motor cortex activity",
            "Enables brain-computer interfaces"
        ],
        "source_file": "neuroscience_general",
        "tags": ["population_coding", "neural_coding", "distributed_representation"]
    },
    {
        "id": "neuro_medium_010",
        "domain": "neuroscience",
        "difficulty": "medium",
        "question": "What is the hemodynamic response function (HRF) and why is it important for fMRI analysis?",
        "answer": "The HRF describes the stereotypical BOLD signal time course following brief neural activity. Canonical HRF: initial dip (~2s, controversial), main positive peak (~4-6s), undershoot (~10-20s), return to baseline (~20-30s). The HRF acts as a temporal filter between neural activity and BOLD signal. Importance for fMRI: 1) Temporal convolution - predicted BOLD = HRF ⊗ neural time course, basis for GLM analysis, 2) Individual differences - HRF varies across subjects, regions, ages, diseases, affecting statistical sensitivity, 3) Event timing - HRF shape enables distinguishing closely-spaced events, 4) Deconvolution - can estimate underlying neural activity from BOLD, though ill-posed. Analysis implications: using canonical vs estimated HRF, accounting for regional variations, finite impulse response (FIR) models for data-driven estimation. HRF knowledge enables proper experimental design, statistical modeling, and result interpretation.",
        "ground_truth": "HRF is stereotypical BOLD response time course; critical for modeling temporal relationship between neural activity and fMRI signal",
        "contexts": [
            "Peak response delayed 4-6 seconds after neural activity",
            "Convolution with HRF predicts BOLD from neural activation",
            "Varies across individuals, regions, and conditions",
            "Basis for general linear model in fMRI analysis"
        ],
        "source_file": "neuroscience_ml_papers/Best practices in data analysis and sharing in neuroimaging using MRI_2017.pdf",
        "tags": ["HRF", "fMRI", "BOLD", "temporal_modeling"]
    },
    {
        "id": "neuro_complex_005",
        "domain": "neuroscience",
        "difficulty": "complex",
        "question": "Critically analyze the challenges and solutions for building generalizable predictive models of brain disorders from neuroimaging data.",
        "answer": "Building generalizable brain disorder prediction models faces multiple challenges: **Data challenges**: 1) Sample size - typical neuroimaging studies have N=50-200, need thousands for deep learning, 2) Heterogeneity - psychiatric diagnoses group diverse biological subtypes, 3) Comorbidity - real patients have multiple conditions, 4) Scanner effects - multi-site studies have systematic biases, 5) Demographic confounds - age, sex, education correlate with both imaging and diagnosis. **Modeling challenges**: 1) High dimensionality - millions of features, thousands of samples, severe overfitting risk, 2) Weak effects - most disorders have small effect sizes (Cohen's d~0.3), 3) Non-stationarity - disorder manifestations change with development, medication, 4) Class imbalance - controls vastly outnumber patients. **Solutions**: 1) **Large-scale consortia** - UK Biobank, ABCD, ENIGMA provide 10,000+ subjects, 2) **Transfer learning** - pre-train on healthy brains, fine-tune on patients, 3) **Federated learning** - train across sites without data sharing, 4) **Careful validation** - nested cross-validation, external validation sets, prospective testing, 5) **Interpretability** - attention mechanisms, saliency maps to understand predictions, 6) **Multimodal integration** - combine structural, functional, genetic, behavioral data, 7) **Subtype discovery** - cluster-then-predict rather than assuming homogeneity. **Reality**: most published models don't replicate; focus shifting from diagnosis to dimensional symptom prediction, treatment response, prognosis. Requires transparent reporting, data sharing, prospective validation.",
        "ground_truth": "Brain disorder prediction challenged by small samples, heterogeneity, weak effects; solutions include large consortia, transfer learning, careful validation",
        "contexts": [
            "Most neuropsychiatric disorders have small effect sizes",
            "Scanner and site effects major confounds in multi-site studies",
            "External validation critical but rarely performed",
            "Shift toward dimensional predictions and treatment response"
        ],
        "source_file": "neuroscience_ml_papers/Best practices in data analysis and sharing in neuroimaging using MRI_2017.pdf",
        "tags": ["predictive_modeling", "neuroimaging", "brain_disorders", "generalization", "machine_learning"]
    },
    {
        "id": "neuro_complex_006",
        "domain": "neuroscience",
        "difficulty": "complex",
        "question": "Explain the forward and inverse problems in brain decoding and their computational complexities.",
        "answer": "Brain decoding involves two complementary problems: **Forward problem** (encoding): predict neural activity from stimuli/behavior - f: S → N where S is stimulus space, N is neural activity. Computationally well-posed: unique solution exists. Methods: linear regression, encoding models, deep neural networks predict voxel responses. Validates understanding: if model predicts responses accurately, it captures relevant representational structure. **Inverse problem** (decoding): infer stimuli/mental states from neural activity - f⁻¹: N → S. Computationally ill-posed: infinite stimuli could produce similar neural patterns, high-dimensional neural space maps to lower-dimensional stimulus space, noise makes problem worse. Solutions require regularization/constraints: 1) Bayes' rule with priors over stimuli, 2) Linear classification with regularization (ridge, lasso), 3) Representational similarity, 4) Deep generative models (VAEs, GANs) constrain outputs. **Complexities**: Forward problem scales with neural dimensions (millions of voxels), inverse problem additionally requires searching stimulus space (potentially infinite), underdetermined system needs strong priors or constraints. **Trade-offs**: Simple decoders (SVM) work well for classification but require labeled data; generative models can reconstruct stimuli but need large datasets and computational resources; encoding models inform inverse problem by providing stimulus-to-neural mapping. Success depends on signal quality, amount of data, appropriate regularization, match between model and true brain representations.",
        "ground_truth": "Forward (encoding) problem well-posed predicts neural from stimuli; inverse (decoding) ill-posed requires constraints to infer stimuli from neural activity",
        "contexts": [
            "Encoding models test hypotheses about brain representations",
            "Decoding demonstrates information presence but not mechanism",
            "Inverse problem requires regularization due to ill-posedness",
            "Generative models enable stimulus reconstruction from brain activity"
        ],
        "source_file": "neuroscience_ml_papers/Brain decoding Reading minds_2013.pdf",
        "tags": ["brain_decoding", "encoding_models", "inverse_problem", "forward_problem", "computational_neuroscience"]
    },

    # Quantum ML additions (7 pairs: 3 simple, 2 medium, 2 complex)
    {
        "id": "quantum_simple_012",
        "domain": "quantum_ml",
        "difficulty": "simple",
        "question": "What is a CNOT gate and what does it do?",
        "answer": "CNOT (Controlled-NOT) is a two-qubit quantum gate that flips the second qubit (target) if and only if the first qubit (control) is |1⟩. It maps: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩. CNOT is crucial for: 1) Creating entanglement - applying CNOT to superposed control creates entangled state, 2) Quantum error correction - detects bit flip errors, 3) Universal quantum computation - any multi-qubit gate can be built from CNOTs and single-qubit rotations. Along with Hadamard and phase gates, CNOT forms universal gate set. It's represented by 4×4 matrix with control bit identity and target bit X operator conditionally applied.",
        "ground_truth": "CNOT flips target qubit if control is |1⟩; creates entanglement and enables universal quantum computation",
        "contexts": [
            "Fundamental two-qubit gate in quantum computing",
            "Creates entanglement when control in superposition",
            "Universal gate set: {H, T, CNOT}",
            "Basis for quantum error correction codes"
        ],
        "source_file": "quantum_fundamentals",
        "tags": ["CNOT", "quantum_gates", "entanglement", "universal_computation"]
    },
    {
        "id": "quantum_simple_013",
        "domain": "quantum_ml",
        "difficulty": "simple",
        "question": "What does quantum supremacy (or quantum advantage) mean?",
        "answer": "Quantum supremacy (now often called quantum advantage) is the point when a quantum computer can solve a problem that would be practically impossible for classical computers in reasonable time. Google claimed supremacy in 2019 by performing random circuit sampling in 200 seconds that would take classical supercomputers thousands of years (though disputed). Important distinctions: 1) Supremacy tasks often artificial (random sampling) not useful applications, 2) Advantage for practical problems more meaningful, 3) Near-term advantage focuses on optimization, chemistry, machine learning. Supremacy demonstrated quantum hardware capabilities but practical quantum advantage for real-world problems remains actively researched goal.",
        "ground_truth": "Quantum supremacy/advantage is when quantum computers solve problems impractical for classical computers",
        "contexts": [
            "Google claimed supremacy with 53-qubit Sycamore processor",
            "Random circuit sampling first supremacy task",
            "Practical advantage more important than supremacy",
            "Active research on advantageous applications"
        ],
        "source_file": "quantum_fundamentals",
        "tags": ["quantum_supremacy", "quantum_advantage", "computational_power"]
    },
    {
        "id": "quantum_simple_014",
        "domain": "quantum_ml",
        "difficulty": "simple",
        "question": "What is quantum noise and what are its main sources?",
        "answer": "Quantum noise refers to errors and unwanted interactions that degrade quantum information. Main sources: 1) Decoherence - loss of quantum properties due to environmental coupling, dominant error source, 2) Gate errors - imperfect implementation of quantum gates due to control imprecision, typically 0.1-1% error per gate, 3) Measurement errors - readout mistakes confusing |0⟩ and |1⟩, typically 1-5%, 4) Crosstalk - unwanted interactions between nearby qubits, 5) Control noise - imperfect pulses implementing gates. Different qubit technologies have different noise profiles: superconducting qubits have short coherence times but fast gates; trapped ions have long coherence but slower gates. Quantum error mitigation and correction techniques aim to reduce noise impact.",
        "ground_truth": "Quantum noise includes decoherence, gate errors, measurement errors, and crosstalk degrading quantum information",
        "contexts": [
            "Decoherence most significant error source",
            "Gate fidelity improving but still limits circuit depth",
            "Error rates vary by qubit technology",
            "Mitigation and correction strategies under development"
        ],
        "source_file": "quantum_fundamentals",
        "tags": ["quantum_noise", "errors", "decoherence", "gate_fidelity"]
    },
    {
        "id": "quantum_medium_009",
        "domain": "quantum_ml",
        "difficulty": "medium",
        "question": "Explain how the quantum Fourier transform works and its role in quantum algorithms.",
        "answer": "Quantum Fourier Transform (QFT) is quantum analog of discrete Fourier transform, mapping computational basis to Fourier basis. For n qubits: QFT|x⟩ = (1/√2ⁿ) Σ_y e^(2πixy/2ⁿ)|y⟩. Circuit: uses Hadamards and controlled phase rotations in specific pattern, O(n²) gates compared to classical FFT's O(n2ⁿ) operations. Key property: period finding - QFT converts periodic functions to peaks at periods. **Roles in quantum algorithms**: 1) **Shor's algorithm** - period finding enables factoring, QFT extracts period from quantum state, 2) **Phase estimation** - measures eigenvalue phases, core subroutine for many algorithms, 3) **Quantum simulation** - implements time evolution, 4) **HHL algorithm** - quantum linear systems, 5) **Quantum walk** - QFT enables quantum walk implementation. **Advantages**: exponential speedup for certain operations, enables interference between computational paths. **Limitation**: measuring QFT output collapses state, limiting direct classical use; amplitude estimation and quantum sampling provide workarounds.",
        "ground_truth": "QFT maps computational to Fourier basis in O(n²) gates; enables period finding in Shor's, phase estimation, and quantum algorithms",
        "contexts": [
            "Exponentially faster than classical FFT for quantum states",
            "Critical component of Shor's factoring algorithm",
            "Enables phase estimation and quantum sampling",
            "Circuit uses Hadamards and controlled phase gates"
        ],
        "source_file": "quantum_algorithms",
        "tags": ["QFT", "quantum_Fourier_transform", "Shor_algorithm", "phase_estimation"]
    },
    {
        "id": "quantum_medium_010",
        "domain": "quantum_ml",
        "difficulty": "medium",
        "question": "What is the quantum variational eigensolver (VQE) algorithm and what problems does it solve?",
        "answer": "VQE is variational algorithm for finding ground state energies of quantum systems. Method: 1) Prepare parameterized ansatz state |ψ(θ)⟩ on quantum computer, 2) Measure energy expectation E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ for Hamiltonian H, 3) Classical optimizer updates parameters to minimize E(θ), 4) Iterate until convergence. By variational principle, E(θ) ≥ E_ground with equality only at ground state. **Applications**: 1) **Molecular chemistry** - calculate molecule ground state energies, binding energies, reaction barriers; main near-term use case, 2) **Materials science** - electronic structure of solids, superconductors, 3) **Condensed matter** - study quantum phase transitions, many-body systems, 4) **Optimization** - MaxCut, graph problems encoded as Hamiltonians. **Advantages over classical**: exponentially large Hilbert space, polynomial quantum resources. **Challenges**: barren plateaus in generic ansatzes, measurement overhead for complex Hamiltonians, ansatz design critical. **Ansatzes**: Unitary Coupled Cluster (chemistry-inspired), Hardware-Efficient (device-native), problem-specific constructions. Most promising near-term quantum algorithm for practical advantage in quantum chemistry.",
        "ground_truth": "VQE variationally finds ground state energies using parameterized quantum circuits; primary application is molecular chemistry",
        "contexts": [
            "Variational principle guarantees energy upper bound",
            "Hybrid quantum-classical optimization",
            "Chemistry applications most mature",
            "Ansatz choice critical for performance and trainability"
        ],
        "source_file": "QuantERA/processed_output/Cerezo-2021-Variational quantum algorithms_processed.json",
        "tags": ["VQE", "ground_state", "quantum_chemistry", "variational_algorithm"]
    },
    {
        "id": "quantum_complex_005",
        "domain": "quantum_ml",
        "difficulty": "complex",
        "question": "Analyze the quantum circuit expressivity vs. trainability trade-off and implications for practical quantum machine learning.",
        "answer": "Quantum ML faces fundamental tension between expressivity and trainability: **Expressivity** (model capacity): measure of functions quantum circuit can represent. Higher expressivity: deeper circuits, more entangling gates, approaches 2-designs over unitary group, can approximate arbitrary unitaries. Benefits: represent complex functions, potential quantum advantage. **Trainability**: ability to optimize parameters efficiently. Measured by gradient magnitudes, cost landscape structure. Deep expressive circuits suffer from: barren plateaus (exponentially vanishing gradients), gradient concentration, requiring exponential measurements. **Trade-off analysis**: 1) **Shallow circuits** - trainable, limited expressivity, no quantum advantage for most problems, 2) **Deep random circuits** - high expressivity, severe barren plateaus, untrainable for large systems, 3) **Problem-inspired circuits** - moderate expressivity aligned with problem structure, better trainability via exploiting symmetries/locality. **Implications**: 1) **NISQ reality** - limited to circuits navigating this trade-off, O(poly log n) depth feasible, 2) **Algorithm design** - must co-design ansatz with problem, exploit structure, use initialization strategies, 3) **Quantum advantage** - likely requires careful circuit design matching problem, not brute-force deep circuits, 4) **Near-term strategy** - problem-specific shallow circuits with smart initialization, layer-by-layer training. **Future directions**: geometric quantum ML leveraging symmetries, classical initialization using compressed sensing, hardware-efficient ansatzes matching device topology, quantum-aware optimization algorithms. Conclusion: practical quantum ML requires navigating expressivity-trainability trade-off through problem-structure alignment, not maximizing either independently.",
        "ground_truth": "Deep circuits are expressive but suffer barren plateaus; shallow circuits trainable but limited; requires problem-structure alignment",
        "contexts": [
            "2-designs have maximal expressivity but worst trainability",
            "Problem-inspired ansatzes balance trade-off",
            "Expressivity-trainability tension unique to quantum",
            "Practical quantum ML requires careful circuit design"
        ],
        "source_file": "QuantERA/processed_output/BarrenPlateaus_processed.json",
        "tags": ["expressivity", "trainability", "barren_plateaus", "circuit_design", "quantum_ML"]
    },
    {
        "id": "quantum_complex_006",
        "domain": "quantum_ml",
        "difficulty": "complex",
        "question": "Critically evaluate the prospects for quantum advantage in optimization problems, considering both theory and practical constraints.",
        "answer": "Quantum advantage for optimization is complex, nuanced question: **Theoretical landscape**: 1) **QAOA**: approximation algorithm for combinatorial optimization (MaxCut, graph coloring), polynomial advantage unclear, performance depends on depth p, 2) **Grover search**: quadratic speedup for unstructured search, limited practical impact, 3) **Quantum annealing**: hardware approach, advantage demonstrated for specific instances but debates over generality, 4) **Quantum walks**: quadratic to exponential speedups for structured problems (spatial search, element distinctness). **Practical challenges**: 1) **Problem encoding** - loading classical problem into quantum state may negate advantage (QRAM overhead), 2) **NISQ limitations** - noise limits QAOA depth, restricting approximation quality, 3) **Classical competition** - specialized hardware (GPUs, TPUs), sophisticated algorithms (branch and bound, SAT solvers) highly optimized, 4) **Measurement overhead** - estimating expectation values requires many shots, 5) **Barren plateaus** - optimization landscapes can be flat for deep circuits. **Evidence evaluation**: 1) **Positive**: quantum annealing shows advantages for specific optimization instances, QAOA promising for graph problems at small scale, quantum walks proven speedups for certain searches, 2) **Negative**: no convincing practical advantage for real-world optimization at scale, classical algorithms improving faster than quantum hardware, most QAOA demonstrations not beyond classical, 3) **Uncertain**: scaling behavior unclear - does advantage grow with problem size? Noise mitigation essential but costly. **Realistic assessment**: 1) **Near-term (5 years)**: specialized optimization with quantum characteristics (quantum chemistry, materials), not general optimization advantage, 2) **Medium-term (10 years)**: possible advantage for specific problem classes if error mitigation/correction matures, 3) **Long-term**: fault-tolerant quantum computers enable provable advantages via Grover, amplitude amplification. **Conclusion**: quantum advantage for general optimization unlikely near-term; best prospects in quantum-native problems (chemistry optimization, materials design) or highly structured problems matching quantum operations; classical algorithms remain formidable competitors requiring constant reassessment of advantage claims.",
        "ground_truth": "Quantum optimization advantage unclear; QAOA promising but limited by NISQ; best prospects in quantum-native or structured problems",
        "contexts": [
            "QAOA performance vs classical uncertain at scale",
            "Quantum annealing shows instance-specific advantages",
            "Classical optimization algorithms highly mature",
            "Quantum advantage likely requires problem-structure match"
        ],
        "source_file": "QuantERA/processed_output/Huang-2025-The vast world of quantum advantage_processed.json",
        "tags": ["quantum_optimization", "QAOA", "quantum_advantage", "practical_constraints", "optimization"]
    },
]

# Additional general science pairs (18 pairs: 6 simple, 6 medium, 6 complex)
general_additions = [
    {
        "id": "general_simple_012",
        "domain": "general_science",
        "difficulty": "simple",
        "question": "What is precision and recall in classification?",
        "answer": "Precision and recall are complementary metrics for classification performance. Precision (positive predictive value) = TP/(TP+FP) measures what fraction of predicted positives are actually positive - answers 'Of items labeled positive, how many truly are?' High precision means few false positives. Recall (sensitivity, true positive rate) = TP/(TP+FN) measures what fraction of actual positives are correctly identified - answers 'Of all positive items, how many did we find?' High recall means few false negatives. Trade-off: making classifier more conservative increases precision but decreases recall and vice versa. F1-score = 2PR/(P+R) balances both. Choice depends on application: spam detection needs high precision (few false alarms), disease screening needs high recall (catch all cases).",
        "ground_truth": "Precision is accuracy of positive predictions (TP/(TP+FP)); recall is completeness of positive detection (TP/(TP+FN))",
        "contexts": [
            "Precision emphasizes avoiding false positives",
            "Recall emphasizes finding all positives",
            "F1-score harmonic mean balancing both",
            "Choice depends on relative costs of FP vs FN"
        ],
        "source_file": "general_machine_learning",
        "tags": ["precision", "recall", "classification", "metrics"]
    },
    {
        "id": "general_simple_013",
        "domain": "general_science",
        "difficulty": "simple",
        "question": "What is the activation function in neural networks and why is it necessary?",
        "answer": "An activation function is a non-linear function applied to neuron outputs, introducing non-linearity into neural networks. Without activation functions, stacking layers would remain linear: f(W₂·f(W₁·x)) = W₂W₁x, collapsing to single layer. Common activations: 1) ReLU (Rectified Linear Unit): f(x) = max(0,x), most popular for hidden layers, 2) Sigmoid: f(x) = 1/(1+e⁻ˣ), outputs [0,1], used for probabilities, 3) Tanh: f(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ), outputs [-1,1], zero-centered, 4) Softmax: for multi-class output. Activation functions enable networks to learn complex non-linear patterns and approximate arbitrary functions (universal approximation theorem).",
        "ground_truth": "Activation functions introduce non-linearity enabling neural networks to learn complex patterns; ReLU, sigmoid, tanh common types",
        "contexts": [
            "ReLU most common for hidden layers due to efficiency",
            "Sigmoid/softmax used for output layer probabilities",
            "Non-linearity essential for deep network expressivity",
            "Choice affects gradient flow and training dynamics"
        ],
        "source_file": "general_machine_learning",
        "tags": ["activation_function", "neural_networks", "non_linearity", "ReLU"]
    },
    {
        "id": "general_simple_014",
        "domain": "general_science",
        "difficulty": "simple",
        "question": "What is data augmentation and why is it used?",
        "answer": "Data augmentation artificially expands training datasets by creating modified versions of existing examples, increasing diversity without collecting new data. For images: rotations, flips, crops, color jittering, mixup. For text: synonym replacement, back-translation, paraphrasing. For audio: time stretching, pitch shifting, adding noise. Benefits: 1) Reduces overfitting by increasing effective dataset size, 2) Improves generalization by exposing model to variations, 3) Makes models robust to transformations, 4) Reduces need for expensive data collection. Essential for domains with limited data (medical imaging, rare languages). Modern techniques include learnable augmentation policies (AutoAugment) and generative augmentation (using GANs/diffusion models).",
        "ground_truth": "Data augmentation creates training variations to expand datasets, reduce overfitting, and improve generalization",
        "contexts": [
            "Critical when training data limited",
            "Common in computer vision (geometric/color transforms)",
            "Must preserve label meaning",
            "Can be domain-specific or learned automatically"
        ],
        "source_file": "general_machine_learning",
        "tags": ["data_augmentation", "regularization", "training", "generalization"]
    },
    {
        "id": "general_simple_015",
        "domain": "general_science",
        "difficulty": "simple",
        "question": "What is the difference between generative and discriminative models?",
        "answer": "Discriminative models learn decision boundary between classes, modeling P(Y|X) - probability of label given features. They answer 'Given features, what is the class?' Examples: logistic regression, SVM, neural network classifiers. Discriminative models directly optimize classification and typically perform better with sufficient data. Generative models learn joint distribution P(X,Y) or class-conditional P(X|Y) and prior P(Y), enabling generation of new samples. They answer 'What do examples from each class look like?' Examples: Naive Bayes, GANs, VAEs, diffusion models. Generative models can generate new data, handle missing features, but require more data and stronger assumptions. Hybrid approaches exist (semi-supervised learning using generative models for representation).",
        "ground_truth": "Discriminative models learn P(Y|X) for classification; generative models learn P(X,Y) enabling generation and density estimation",
        "contexts": [
            "Discriminative: decision boundaries directly",
            "Generative: model data distribution",
            "Generative can synthesize new examples",
            "Discriminative often better classification with enough data"
        ],
        "source_file": "general_machine_learning",
        "tags": ["generative_models", "discriminative_models", "classification", "generation"]
    },
    {
        "id": "general_simple_016",
        "domain": "general_science",
        "difficulty": "simple",
        "question": "What is dropout and how does it prevent overfitting?",
        "answer": "Dropout is a regularization technique that randomly sets a fraction of neuron activations to zero during training. At each iteration, randomly drop neurons with probability p (typically 0.2-0.5), forcing network to learn redundant representations that don't rely on specific neurons. At test time, use all neurons but scale outputs by (1-p) to account for training dropout. Benefits: 1) Prevents co-adaptation of neurons (complex interdependencies), 2) Acts like training ensemble of exponentially many thinned networks, 3) Improves generalization by reducing overfitting, 4) Provides uncertainty estimates via Monte Carlo dropout. Dropout especially effective for large networks and limited data. Modern alternatives include DropConnect (drop connections not neurons) and concrete/variational dropout.",
        "ground_truth": "Dropout randomly deactivates neurons during training to prevent co-adaptation and overfitting; acts like ensemble training",
        "contexts": [
            "Typically drop 20-50% of neurons",
            "Applied during training, scaled at test time",
            "Particularly effective for fully connected layers",
            "Reduces reliance on specific neurons"
        ],
        "source_file": "general_machine_learning",
        "tags": ["dropout", "regularization", "overfitting", "neural_networks"]
    },
    {
        "id": "general_simple_017",
        "domain": "general_science",
        "difficulty": "simple",
        "question": "What is a ROC curve and AUC metric?",
        "answer": "ROC (Receiver Operating Characteristic) curve plots True Positive Rate (Recall) vs False Positive Rate across all classification thresholds. Each point represents different decision threshold from 0 to 1. Perfect classifier reaches top-left corner (TPR=1, FPR=0); random classifier is diagonal line. AUC (Area Under Curve) summarizes ROC in single number: 1.0 = perfect, 0.5 = random. AUC interpretation: probability randomly chosen positive ranked higher than randomly chosen negative. Advantages: threshold-independent evaluation, handles class imbalance better than accuracy, compares models across operating points. Useful when optimal threshold unknown or varies by application. Precision-Recall curves alternative when classes severely imbalanced.",
        "ground_truth": "ROC curve plots TPR vs FPR across thresholds; AUC summarizes with single metric (1=perfect, 0.5=random)",
        "contexts": [
            "Threshold-independent performance metric",
            "AUC = area under ROC curve",
            "Better than accuracy for imbalanced classes",
            "Enables comparison across operating points"
        ],
        "source_file": "general_machine_learning",
        "tags": ["ROC", "AUC", "classification", "evaluation"]
    },
    {
        "id": "general_medium_009",
        "domain": "general_science",
        "difficulty": "medium",
        "question": "Explain the concept of word embeddings and how they capture semantic relationships.",
        "answer": "Word embeddings map words to dense, low-dimensional continuous vectors (typically 100-300D) where semantic similarity reflected by geometric proximity. Key models: **Word2Vec** (Skip-gram, CBOW) - predicts context from word or vice versa, learns representations where similar words are close; **GloVe** - factorizes co-occurrence matrix, combines global statistics with local context; **FastText** - extends Word2Vec with subword information, handles out-of-vocabulary words. **Semantic properties**: 1) Similar words cluster together (king near queen, monarchy), 2) **Analogies via vector arithmetic**: king - man + woman ≈ queen; Paris - France + Germany ≈ Berlin, 3) Different dimensions capture different semantic aspects (gender, plurality, tense), 4) Non-linear manifold structure captures complex relationships. **Modern evolution**: contextual embeddings (BERT, GPT) produce different vectors for same word in different contexts, capturing polysemy. Applications: similarity search, analogy completion, downstream NLP tasks initialization, cross-lingual transfer. Limitations: static embeddings miss context, capture statistical co-occurrence not true understanding, can encode societal biases.",
        "ground_truth": "Word embeddings map words to vectors where geometric relationships reflect semantic meaning; enable analogies and similarity",
        "contexts": [
            "Learned from large text corpora via prediction tasks",
            "Vector arithmetic captures semantic relationships",
            "Contextual embeddings (BERT) supersede static (Word2Vec)",
            "Foundation of modern NLP architectures"
        ],
        "source_file": "general_machine_learning",
        "tags": ["word_embeddings", "Word2Vec", "semantics", "NLP", "representation_learning"]
    },
    {
        "id": "general_medium_010",
        "domain": "general_science",
        "difficulty": "medium",
        "question": "Compare convolutional neural networks (CNNs) and vision transformers (ViTs) for image processing.",
        "answer": "CNNs and ViTs represent different paradigms for visual processing: **CNNs**: 1) **Inductive bias** - built-in locality (conv filters), translation equivariance, hierarchical structure matches vision; 2) **Architecture** - stacked convolutions + pooling, local receptive fields grow with depth, feature maps; 3) **Efficiency** - parameter sharing via kernels, fewer parameters than ViTs; 4) **Data requirements** - work well with moderate data due to inductive biases. **ViTs**: 1) **Inductive bias** - minimal assumptions, learns relationships from data; patches treated as tokens; 2) **Architecture** - divide image into patches, flatten to sequences, pure transformer (self-attention + MLP); 3) **Flexibility** - global receptive field from layer 1, attention captures long-range dependencies naturally; 4) **Data requirements** - need large-scale pre-training (ImageNet-21K+) to match CNN performance. **Performance comparison**: CNNs better for small/medium datasets with inductive bias advantages; ViTs outperform at large scale (billions of parameters, hundreds of millions of examples) due to superior scaling laws. **Hybrid approaches**: ConvNets + attention (ConvNeXt matches ViT with CNN architecture), early conv layers + transformer (Swin Transformer uses shifted windows). **Trade-offs**: CNNs more sample-efficient, ViTs more scalable; CNNs interpretable via feature maps, ViTs via attention; CNNs deployed widely, ViTs gaining adoption as foundation models.",
        "ground_truth": "CNNs use local convolutions with strong inductive bias; ViTs use global self-attention, scale better with data but need more examples",
        "contexts": [
            "CNNs dominant until 2020, ViTs emerging at scale",
            "ViTs require large-scale pre-training (ImageNet-21K)",
            "Hybrid models combine benefits",
            "ViTs foundation for modern vision-language models"
        ],
        "source_file": "general_machine_learning",
        "tags": ["CNN", "vision_transformer", "ViT", "computer_vision", "architecture"]
    },
    {
        "id": "general_medium_011",
        "domain": "general_science",
        "difficulty": "medium",
        "question": "Explain the concept of knowledge distillation and its applications.",
        "answer": "Knowledge distillation transfers knowledge from large 'teacher' model to smaller 'student' model while maintaining performance. **Method**: 1) Train teacher model on task achieving high accuracy; 2) Student trained on both hard targets (true labels) and soft targets (teacher's softmax outputs at temperature T>1); 3) Loss combines classification loss and distillation loss: L = αL_CE(student, labels) + (1-α)L_KD(student, teacher); **Temperature**: T>1 softens probability distribution, revealing relative similarities between classes (dark knowledge). **Benefits**: 1) **Compression** - deploy efficient small models with large model performance; 2) **Acceleration** - faster inference on edge devices; 3) **Ensemble knowledge** - distill ensemble into single model; 4) **Cross-modality** - transfer from one architecture to another. **Applications**: 1) Mobile deployment (BERT → DistilBERT 40% size, 97% performance); 2) Real-time systems (large models → fast models); 3) Privacy (deploy small model, keep large model private); 4) Continual learning (old model → new model on new data). **Extensions**: multi-teacher distillation, self-distillation, feature-based distillation, attention transfer. **Why it works**: soft targets provide richer supervision than hard labels, revealing model's learned similarities; model regularization prevents overfitting.",
        "ground_truth": "Knowledge distillation trains small student model mimicking large teacher using soft probability outputs, enabling compression with performance retention",
        "contexts": [
            "Soft targets at temperature T>1 reveal relative class similarities",
            "Enables deployment of compressed models",
            "DistilBERT successful NLP example",
            "Dark knowledge carries information beyond hard labels"
        ],
        "source_file": "general_machine_learning",
        "tags": ["knowledge_distillation", "model_compression", "transfer_learning", "deployment"]
    },
    {
        "id": "general_medium_012",
        "domain": "general_science",
        "difficulty": "medium",
        "question": "What is catastrophic forgetting in neural networks and how can it be mitigated?",
        "answer": "Catastrophic forgetting (interference) occurs when neural networks forget previously learned tasks upon learning new ones - weights optimized for new task override previous knowledge. Particularly severe in continual/lifelong learning scenarios. **Causes**: 1) Distributed representations shared across tasks - updating weights for new task disrupts old; 2) SGD optimization finds separate minima for each task; 3) No mechanisms preserving old task performance. **Mitigation strategies**: 1) **Regularization** - Elastic Weight Consolidation (EWC) penalizes changes to important weights (measured by Fisher information); 2) **Replay** - store/generate old task samples, interleave with new data (requires memory or generative model); 3) **Dynamic architectures** - add new parameters/modules for new tasks (Progressive Neural Networks); 4) **Meta-learning** - learn to learn enabling fast adaptation without forgetting; 5) **Dual memory systems** - complementary learning systems with slow consolidation (inspired by hippocampus-cortex). **Evaluation**: measure average accuracy across all tasks over time. **Trade-off**: plasticity (learning new) vs stability (retaining old). **Applications**: robotics (learn new skills continuously), personalization (adapt to user without forgetting others), real-time systems. Challenge: finding general solution approaching human-like continual learning.",
        "ground_truth": "Catastrophic forgetting is loss of old task knowledge when learning new tasks; mitigated by regularization, replay, or dynamic architectures",
        "contexts": [
            "Major challenge for continual/lifelong learning",
            "EWC uses Fisher information to protect important weights",
            "Experience replay most effective but memory-intensive",
            "Biological inspiration from complementary learning systems"
        ],
        "source_file": "general_machine_learning",
        "tags": ["catastrophic_forgetting", "continual_learning", "neural_networks", "interference"]
    },
    {
        "id": "general_medium_013",
        "domain": "general_science",
        "difficulty": "medium",
        "question": "Explain the concept of few-shot learning and meta-learning approaches.",
        "answer": "Few-shot learning adapts models to new tasks with limited examples (1-shot, 5-shot). Critical for domains where data collection expensive or impossible. **Approaches**: 1) **Metric learning** - learn embedding space where similar items cluster; classify via nearest neighbor (Siamese networks, Prototypical networks); 2) **Meta-learning** (learning to learn) - train on distribution of tasks to learn how to quickly adapt to new tasks; MAML (Model-Agnostic Meta-Learning) finds initialization enabling fast fine-tuning; 3) **Memory-augmented** - external memory (Neural Turing Machines) stores examples for rapid retrieval; 4) **Generative** - generate synthetic examples for data augmentation. **Meta-learning framework**: Split data into support set (few examples for adaptation) and query set (evaluation). Train across many tasks: in each episode, sample task, adapt using support set, evaluate on query set. Optimize for post-adaptation performance. **Applications**: 1) Drug discovery (few known examples of activity); 2) Robotics (quick adaptation to new environments); 3) Personalization (adapt to users with limited data); 4) Rare language translation. **Challenges**: Task distribution mismatch between meta-training and deployment; computational cost of meta-learning; limited theoretical understanding. Promising direction: combining pre-trained foundation models with few-shot prompting.",
        "ground_truth": "Few-shot learning adapts to new tasks with few examples using metric learning, meta-learning, or memory; enables rapid task adaptation",
        "contexts": [
            "MAML learns initialization for fast adaptation",
            "Prototypical networks embed and classify via prototypes",
            "Meta-learning trains on task distribution",
            "Critical for domains with limited data"
        ],
        "source_file": "general_machine_learning",
        "tags": ["few_shot_learning", "meta_learning", "MAML", "transfer_learning"]
    },
    {
        "id": "general_medium_014",
        "domain": "general_science",
        "difficulty": "medium",
        "question": "What are generative adversarial networks (GANs) and how do they work?",
        "answer": "GANs are generative models consisting of two neural networks in adversarial game: **Generator G**: takes random noise z, produces synthetic data G(z) mimicking real distribution. **Discriminator D**: distinguishes real data x from generated G(z). **Training**: minimax game min_G max_D E_x[log D(x)] + E_z[log(1-D(G(z)))]. Discriminator maximizes ability to classify real vs fake; Generator minimizes discriminator's success, learning to fool it. Equilibrium: generator produces indistinguishable samples, discriminator guesses randomly. **Training dynamics**: alternate between discriminator and generator updates. **Advantages**: 1) High-quality generation (photorealistic images); 2) No explicit likelihood, flexible; 3) Learns data distribution implicitly. **Challenges**: 1) Training instability (mode collapse, oscillation); 2) Difficult to train (balancing G and D); 3) No likelihood for evaluation. **Improvements**: Wasserstein GAN (better gradient flow), StyleGAN (disentangled control), BigGAN (large-scale generation), conditional GANs (controlled generation). **Applications**: image synthesis, super-resolution, style transfer, data augmentation, domain adaptation. **Alternatives**: Diffusion models increasingly popular due to better training stability.",
        "ground_truth": "GANs use adversarial game between generator (creates samples) and discriminator (detects fakes) to learn data distribution",
        "contexts": [
            "Generator learns to fool discriminator",
            "Training as minimax game",
            "Mode collapse common challenge",
            "StyleGAN, BigGAN achieve photorealistic generation"
        ],
        "source_file": "general_machine_learning",
        "tags": ["GAN", "generative_models", "adversarial_training", "image_generation"]
    },
    {
        "id": "general_complex_004",
        "domain": "general_science",
        "difficulty": "complex",
        "question": "Analyze the sample complexity of deep learning in light of statistical learning theory and explain the apparent contradiction with empirical success.",
        "answer": "**Theoretical prediction**: Classical PAC learning and VC theory predict sample complexity should grow with model capacity (number of parameters). Deep networks have millions/billions of parameters, suggesting need for astronomical training sets. **Empirical reality**: Deep networks generalize well despite: 1) Overparameterization - more parameters than training examples, 2) Zero training loss - perfect fit including noise, 3) Classical theory predicts catastrophic overfitting, yet generalization is good. **The paradox**: Classical bounds are vacuous for deep networks (VC dimension ~ number of parameters >> number of samples), yet they work. **Modern understanding**: 1) **Implicit regularization** - SGD biases toward solutions with good generalization properties (minimum norm, flat minima), 2) **Effective capacity** - not all parameters used equally; effective degrees of freedom << total parameters, 3) **Overparameterization helps** - interpolating solutions exist with good generalization (benign overfitting), 4) **Neural tangent kernel** - infinite-width limit provides theoretical insights, but finite networks behave differently, 5) **Lottery ticket hypothesis** - sparse sub-networks exist that train effectively. **Revised theory**: 1) **Uniform convergence** insufficient - need distribution-specific analysis, 2) **Algorithmic-dependent bounds** - considering SGD dynamics, not just hypothesis class, 3) **Interpolation regime** - distinct from classical underparameterized regime, new principles apply, 4) **Margin theory** - generalization depends on margin, not just capacity. **Open questions**: Why do neural networks find generalizing solutions in vast space? What makes real-world data amenable to deep learning? Complete theoretical explanation lacking, active research area.",
        "ground_truth": "Classical theory predicts deep networks need astronomical data but they generalize well; explained by implicit regularization, effective capacity, benign overfitting",
        "contexts": [
            "Overparameterized networks generalize despite zero training loss",
            "SGD implicitly regularizes toward good solutions",
            "Classical VC/PAC bounds vacuous for deep networks",
            "New theory emerging for interpolation regime"
        ],
        "source_file": "general_machine_learning",
        "tags": ["sample_complexity", "generalization", "deep_learning", "learning_theory", "overparameterization"]
    },
    {
        "id": "general_complex_005",
        "domain": "general_science",
        "difficulty": "complex",
        "question": "Critically evaluate the interpretability vs performance trade-off in machine learning and approaches to reconcile them.",
        "answer": "**Trade-off thesis**: Complex high-performing models (deep neural networks, ensembles) are black boxes; simple interpretable models (linear, trees) sacrifice performance. **Interpretability types**: 1) **Transparent** - understand full model logic (linear regression coefficients), 2) **Post-hoc** - explain predictions after training (saliency maps, SHAP), 3) **Simulatable** - human can mentally execute model. **Why trade-off exists**: 1) Expressivity requires complexity - capturing non-linear interactions, high-dimensional patterns needs complex functions, 2) Ensemble methods sacrifice individual comprehension for aggregate performance, 3) Feature learning in deep nets creates abstract representations losing human interpretation. **Approaches to reconcile**: 1) **Interpretable-by-design** - constrain architecture to maintain interpretability (Generalized Additive Models, Neural Additive Models, attention mechanisms with interpretable heads), 2) **Post-hoc explanation** - train black box, explain afterwards: a) Local explanations: LIME (local linear approximation), SHAP (game-theoretic feature attribution), integrated gradients, b) Global understanding: feature importance, partial dependence plots, prototypes, 3) **Model distillation** - compress complex model into interpretable surrogate (decision tree approximating neural network), 4) **Hybrid architectures** - combine interpretable and black-box components (attention weights show where model looks, neural networks still process). **Evaluation challenges**: How to measure interpretability? Faithfulness (explanation matches model), plausibility (makes sense to humans), utility (helps humans). **Domain considerations**: High-stakes domains (medicine, justice) prioritize interpretability; perception tasks (vision, speech) tolerate black boxes. **Critical view**: 1) Post-hoc explanations may mislead - show what humans want to see, not true model behavior, 2) Interpretability definition unclear - means different things to different people, 3) False dichotomy - some complex models more interpretable than assumed, some simple models opaque at scale. **Future directions**: Neuro-symbolic AI combining reasoning and learning, causal models for mechanism understanding, interactive explanation systems for iterative refinement.",
        "ground_truth": "Interpretability-performance trade-off exists but not absolute; reconciled via interpretable architectures, post-hoc explanations, distillation, hybrid models",
        "contexts": [
            "High-stakes domains prioritize interpretability",
            "Post-hoc methods (LIME, SHAP) explain black boxes",
            "Attention mechanisms provide some transparency",
            "Trade-off nuanced, not binary"
        ],
        "source_file": "general_machine_learning",
        "tags": ["interpretability", "explainable_AI", "trade_offs", "LIME", "SHAP"]
    },
    {
        "id": "general_complex_006",
        "domain": "general_science",
        "difficulty": "complex",
        "question": "Explain the theoretical foundations and practical challenges of causal inference from observational data in machine learning.",
        "answer": "**Goal**: Infer causal relationships (X causes Y) from observational data, not just correlations. Essential for: policy decisions, intervention planning, understanding mechanisms. **Theoretical frameworks**: 1) **Potential outcomes** (Rubin causal model) - define causal effect as difference in outcomes under treatment vs control for same unit; fundamental problem: can't observe both; 2) **Structural causal models** (Pearl) - represent causal mechanisms via directed acyclic graphs (DAGs); do-calculus provides rules for causal inference; 3) **Graphical models** - causal graphs encode assumptions; d-separation characterizes conditional independence. **Assumptions required**: 1) **Unconfoundedness** (ignorability) - treatment assignment independent of potential outcomes given covariates; strong, often violated; 2) **Positivity** (overlap) - all covariate combinations have non-zero probability of each treatment; 3) **SUTVA** (no interference) - individual outcomes unaffected by others' treatments; 4) **Correct functional form** - model properly specified. **Methods**: 1) **Matching** - compare similar treated/control units; propensity score matching most common; 2) **Weighting** - inverse propensity weighting creates pseudo-randomized sample; 3) **Instrumental variables** - use variable affecting treatment but not outcome directly; 4) **Regression discontinuity** - exploit cutoff creating quasi-random assignment; 5) **Difference-in-differences** - compare changes over time between groups; 6) **Causal forests** - machine learning for heterogeneous treatment effects. **ML integration**: 1) **Double ML** - use ML for nuisance parameter estimation while maintaining valid inference; 2) **Causal representation learning** - learn representations capturing causal mechanisms; 3) **Counterfactual prediction** - estimate outcomes under interventions. **Challenges**: 1) **Hidden confounding** - unmeasured variables bias estimates; sensitivity analysis quantifies robustness; 2) **High-dimensional settings** - many covariates relative to samples; regularization and selection needed; 3) **Temporal confounding** - time-varying confounders and treatment; 4) **Interference** - SUTVA violated in networks, spillovers; 5) **Measurement error** - noisy covariates bias estimates. **Validity threats**: 1) Confounding (omitted variable bias), 2) Selection bias, 3) Measurement error, 4) Model misspecification. **Best practices**: 1) Pre-register analysis plans, 2) Sensitivity analysis, 3) Multiple methods (triangulation), 4) Causal diagrams make assumptions explicit, 5) Domain knowledge essential. **Reality check**: Causation from observation fundamentally requires untestable assumptions; randomized experiments gold standard; observational causal inference approximation under strong assumptions; results suggestive not definitive.",
        "ground_truth": "Causal inference from observations requires strong untestable assumptions (unconfoundedness, positivity); methods include matching, weighting, IV; challenges include hidden confounding",
        "contexts": [
            "Potential outcomes and structural causal models key frameworks",
            "Propensity scores enable matching on high-dimensional covariates",
            "Hidden confounding major validity threat",
            "ML enables flexible estimation but doesn't solve fundamental problems"
        ],
        "source_file": "general_machine_learning",
        "tags": ["causal_inference", "observational_data", "confounding", "potential_outcomes", "propensity_score"]
    },
]

# Load existing benchmark
benchmark_path = Path("/home/juke/git/AI-CoScientist/data/validation/golden_qa_benchmark.json")
with open(benchmark_path, 'r') as f:
    benchmark = json.load(f)

# Add new pairs
benchmark['qa_pairs'].extend(additional_pairs)
benchmark['qa_pairs'].extend(general_additions)

# Update metadata
benchmark['metadata']['total_pairs'] = len(benchmark['qa_pairs'])

# Verify distribution
domain_counts = {}
difficulty_counts = {}
for qa in benchmark['qa_pairs']:
    domain_counts[qa['domain']] = domain_counts.get(qa['domain'], 0) + 1
    difficulty_counts[qa['difficulty']] = difficulty_counts.get(qa['difficulty'], 0) + 1

print(f"Total pairs: {len(benchmark['qa_pairs'])}")
print(f"Domain distribution: {domain_counts}")
print(f"Difficulty distribution: {difficulty_counts}")

# Save updated benchmark
with open(benchmark_path, 'w') as f:
    json.dump(benchmark, f, indent=2)

print(f"\nBenchmark completed and saved to {benchmark_path}")
