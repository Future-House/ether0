"""Prompt templates used for problems in the ether0 dataset."""

# ruff: noqa: E501, W505

NAME_IUPAC_PROMPTS = [
    "What are the SMILES from the IUPAC name: {iupac}?",
    "Could you please tell me the SMILES representation for {iupac}?",
    "I have {iupac}. What would its SMILES be?",
    "Provide the SMILES string for the molecule named {iupac}.",
    "Convert this IUPAC name into a SMILES format: {iupac}.",
    "Give me the SMILES notation for the compound {iupac}.",
    "What SMILES corresponds to the IUPAC chemical name {iupac}?",
    "How can I represent {iupac} as a SMILES string?",
    "Generate the SMILES structure for this compound: {iupac}.",
    "If the molecule is called {iupac}, what's the SMILES representation?",
]

NAME_SMILES_PROMPTS = [
    "What is the IUPAC name of this molecule: {smiles}?",
    "Could you please tell me the IUPAC name for the compound represented by the SMILES string: {smiles}?",
    "I have a molecule here with the SMILES notation {smiles}. What would its IUPAC name be?",
    "I'm working with a chemical compound, and its SMILES representation is {smiles}. Can you help me determine its IUPAC name?",
    "What is the correct IUPAC nomenclature for a molecule with the SMILES code {smiles}?",
    "I'm trying to identify a compound. Its SMILES string is {smiles}. What's its IUPAC name?",
    "If I provide you with the SMILES string of a molecule, which is {smiles}, could you generate its IUPAC name for me?",
    "I've encountered a molecule in my research, denoted by the SMILES {smiles}. I'd appreciate it if you could tell me its IUPAC name.",
    "Can you derive the IUPAC name from this SMILES representation: {smiles}?",
    "For a compound with the structural representation given by the SMILES {smiles}, what is the corresponding IUPAC name?",
]


REACTION_PROMPTS = [
    "What is the product of this reaction? {rxn_smiles}",
    "If {rxn_smiles} are allowed to react, what would the resulting molecule be in SMILES format?",
    "Can you predict the outcome of this chemical reaction? {rxn_smiles}",
    "I have a reaction scheme here: {rxn_smiles}. What will be generated as the product?",
    "Assuming the reaction {rxn_smiles} goes to completion, what is the SMILES representation of the product?",
    "In this reaction: {rxn_smiles}, what compound is formed?",
    "Given the reactants and conditions in {rxn_smiles}, what is the expected product?",
    "After performing the reaction {rxn_smiles}, what would I obtain as the main product?",
    "If I mix these reactants under the specified conditions, {rxn_smiles}, what is the SMILES of the product?",
    "Please provide the SMILES string of the product for this reaction: {rxn_smiles}",
]

NAME_REACTION_PROMPTS = [
    "What is the name of this reaction?\n{rxn_smiles}",
    "I have a reaction here with {rxn_smiles}. Can you tell me its commonly used name?",
    "Given the reaction represented by {rxn_smiles}, what is the standard name associated with it?",
    "If I were to describe this reaction, {rxn_smiles}, in a textbook, what reaction name would I use?",
    "I'm trying to identify this reaction: {rxn_smiles}. What is its well-known name?",
    "The reaction {rxn_smiles} is taking place. What's the name of this type of transformation?",
    "I came across this reaction pathway: {rxn_smiles}. Do you know the name it generally goes by?",
    "In the context of organic chemistry, what is the established name for the reaction shown here: {rxn_smiles}?",
    "I'm writing a lab report and need to name this reaction, {rxn_smiles}. What should I call it?",
    "Can you identify the name of the reaction that follows this scheme: {rxn_smiles}?",
]

COMPLETE_MOL_PROMPTS = [
    "I have a partial molecule represented by the SMILES string {smiles}. What is a valid completion of this molecule, providing only the remaining characters in SMILES format?",
    "Given the incomplete SMILES fragment {smiles}, can you suggest a realistic ending to complete the molecule? Please provide only the additional SMILES characters needed.",
    "I'm working with a molecule that's partially described as {smiles}. What sequence of SMILES characters would you add to make it a complete, valid molecule?",
    "The beginning of a molecule's SMILES representation is {smiles}. How would you finish this SMILES string to represent a viable chemical compound? Only provide the continuation of the SMILES.",
    "Imagine you need to complete the SMILES string {smiles}. What's a plausible way to extend it to form a complete molecule, expressed as the remaining SMILES characters?",
    "If I give you the partial SMILES {smiles}, what's a reasonable way to finish it off to create a valid molecule? Respond with just the additional SMILES characters.",
    "I'm trying to construct a molecule, and I have the start of its SMILES: {smiles}. Could you provide a completion for it, ensuring the final molecule is realistic? Only give me the rest of the SMILES string.",
    "Here's a fragment of a SMILES string: {smiles}. What would be a chemically sound way to complete it? Respond with the missing portion of the SMILES representation.",
    "Suppose you have the incomplete molecular structure {smiles} in SMILES. How would you complete it to represent a real molecule, adding only the necessary SMILES characters?",
    "I have an unfinished molecule represented by the SMILES fragment {smiles}. Can you help me complete it by suggesting the remaining SMILES characters needed to make it a valid chemical structure?",
]

MOL_FORMULA_PROMPTS = [
    "A compound with formula {formula} was isolated from {source}. What is a plausible SMILES for it given this organism?",
    "{source} makes a compound with this formula: {formula}. What SMILES structure might correspond to it?",
    "In {source}, I found a substance with formula {formula}. What biosynthetically plausible SMILES might this represent?",
    "Analysis of {source} revealed a compound ({formula}). What SMILES structure aligns with this organism's metabolism?",
    "The organism {source} contains a compound with formula {formula}. What's a likely SMILES based on its biochemistry?",
    "A {formula} compound was extracted from {source}. Based on this organism, what's a probable SMILES structure?",
    "What SMILES could have the formula {formula} and be isolated from {source}?",
    "What would be a biologically relevant SMILES for a {formula} compound isolated from the organism {source}?",
    "The organism {source} produced a compound with formula {formula}, what SMILES structure makes biosynthetic sense?",
    "A {formula} metabolite from {source} was identified. What's a biologically plausible compound for this (as SMILES)?",
]

FUNCTIONAL_GROUP_PROMPTS = [
    "Propose a compound with molecular formula {formula} that contains the following functional groups: {functional_group}.",
    "Suggest a SMILES structure for a molecule with formula {formula} and the following functional groups: {functional_group}.",
    "Given that a compound has formula {formula}, propose SMILES for one that also contains these groups: {functional_group}",
    "Provide a reasonable SMILES for a chemical with molecular formula {formula} and these groups: {functional_group}.",
    "Generate a SMILES representation for a molecule containing groups: {functional_group}. It should also have formula {formula}.",
    "Identify a plausible SMILES for a chemical compound with formula {formula} containing these groups: {functional_group}.",
]

PROPERTY_TRIPLET_PROMPTS = [
    "I have a molecule {smiles1} with a {property} of {value1}. Which of these similar molecules will most likely {change} this property?\n{options}",
    "Given a molecule ({smiles1}) having a {property} of {value1}, select the modified molecule below that would {change} this property significantly:\n{options}",
    "Molecule {smiles1} currently exhibits {property} of {value1}. Which modifications from the list below would effectively {change} it?\n{options}",
    "If molecule {smiles1} has a {property} value of {value1}, which of the following options would best {change} this property?\n{options}",
    "Considering {smiles1} has a measured {property} of {value1}, which candidate modification listed would most effectively {change} this property?\n{options}",
    "Molecule {smiles1} demonstrates a {property} of {value1}. Which similar molecule below is best suited to {change} this characteristic?\n{options}",
    "Given molecule {smiles1} with {property} at {value1}, identify which molecule among the following options would {change} it most effectively:\n{options}",
    "Starting from molecule {smiles1}, which shows a {property} of {value1}, choose the structural change below that would notably {change} this property:\n{options}",
    "The molecule {smiles1} has a {property} of {value1}. Which molecule listed would optimally {change} this value?\n{options}",
    "Given a {property} of {value1} for molecule {smiles1}, pick the best molecule from below to {change} this property:\n{options}",
]

# I have a molecule {smiles1} which is not a blood-brain barrier penetrating. Which of these similar molecules will most likely have this property?\n{options}",
PROPERTY_TRIPLET_PROMPTS_CAT = [
    "I have a molecule {smiles1} which {rel} {property}. Which of these similar molecules will most likely {irel} this property?\n{options}",
    "Given molecule {smiles1} that {rel} {property}, which molecule below is likely to {irel} this property?\n{options}",
    "Molecule {smiles1} currently {rel} {property}. Choose from these similar molecules the one most likely to {irel} this property:\n{options}",
    "Considering {smiles1} {rel} {property}, identify which of the following candidates will most likely {irel} the characteristic:\n{options}",
    "Given that molecule {smiles1} {rel} {property}, select from below the molecule most expected to {irel} this characteristic:\n{options}",
    "Starting from molecule {smiles1} which {rel} {property}, determine which listed molecule is most likely to {irel} this property:\n{options}",
    "If molecule {smiles1} {rel} {property}, which of these related structures will most probably {irel} that property?\n{options}",
    "Given molecule {smiles1} {rel} {property}, select the similar molecule listed below most likely to {irel} this property:\n{options}",
]

# Which of the following options likely is a blood-brain barrier penetrating molecule?
# Which of the following options likely is not a blood-brain barrier penetrating molecule?
# Which of the following molecules is likely to not be blood-brain barrier penetrating?
PROPERTY_PROMPTS_CAT = [
    "Which of the following options likely is{rel} a {property} molecule?\n{options}",
    "Which of the following molecules is likely to{rel} be {property}?\n{options}",
    "Identify the molecule below that likely is{rel} a {property} molecule:\n{options}",
    "From the list below, select the molecule most likely to{rel} be {property}:\n{options}",
    "Choose the molecule from the options below that most probably is{rel} {property}:\n{options}",
    "Among the following, which molecule likely is{rel} considered {property}?\n{options}",
    "Select the molecule below most expected to{rel} have {property} properties:\n{options}",
    "From these molecules, identify the one most likely to{rel} possess {property}:\n{options}",
    "Which candidate below most probably is{rel} classified as a {property} molecule?\n{options}",
]


PROPERTY_PROMPTS = [
    "Which of the following molecules likely has a {property} of {value}?\n{options}",
    "Identify the molecule below expected to have a {property} around {value}:\n{options}",
    "From these options, select the molecule most likely exhibiting {property} of {value}:\n{options}",
    "Determine which of the following molecules likely shows a {property} near {value}:\n{options}",
    "Choose the molecule that would most plausibly have a {property} of {value} from the list below:\n{options}",
    "Among the following, which molecule is predicted to have a {property} close to {value}?\n{options}",
    "Given the choices below, pick the molecule most likely to possess a {property} of {value}:\n{options}",
    "Select the molecule from these candidates that probably has a {property} of {value}:\n{options}",
    "Which molecule listed here is most likely to have a {property} approximately equal to {value}?\n{options}",
    "Identify which of the following molecules will most likely have a {property} of {value}:\n{options}",
]

RETRO_PROMPTS = [
    "Propose a 1-step synthesis for the molecule {smiles} using likely purchasable reactants.",
    "Given the molecule {smiles}, suggest a 1-step synthesis using commercially available starting materials.",
    "What is a plausible 1-step reaction for the molecule {smiles} using common reactants?",
    "Suggest a commercially feasible one-step route to synthesize {smiles}.",
    "Outline a practical single-step synthetic method to prepare the molecule {smiles}.",
    "Design a straightforward 1-step reaction scheme for synthesizing {smiles} using commercially available reagents.",
    "Identify a likely accessible precursor and reaction for a single-step synthesis of {smiles}.",
    "Provide a realistic single-step synthetic pathway to obtain {smiles} from common chemicals.",
    "Propose a viable one-step synthetic route toward the molecule {smiles} starting from purchasable precursors.",
    "Suggest one plausible reaction step to generate {smiles} using standard, commercially sourced reactants.",
]

ORACLE_SOLUBILITY_PROMPTS = {
    "tanimoto": [
        "Propose a small change to {smiles} to {direction} its solubility by about 1 logS.",
        "Suggest a minimal structural modification to {smiles} that would {direction} its solubility by approximately 1 logS unit.",
        "What minor alteration could be made to {smiles} to {direction} its solubility by roughly 1 logS?",
        "Design a small molecular change to {smiles} that would {direction} its solubility by about 1 logS while maintaining overall similarity.",
        "Identify a small structural adjustment to {smiles} that would {direction} its aqueous solubility by approximately 1 logS unit.",
    ],
    "scaffold": [
        "Change {smiles} to {direction} its solubility by about 1 logS, but keep its scaffold",
        "Modify {smiles} to {direction} its solubility by approximately 1 logS while preserving the core scaffold structure.",
        "Suggest alterations to {smiles} that would {direction} its solubility by about 1 logS unit without changing the molecular scaffold.",
        "How could {smiles} be transformed to {direction} its solubility by roughly 1 logS while maintaining its scaffold?",
        "Design a derivative of {smiles} with {direction}d solubility (by about 1 logS) that retains the same molecular scaffold.",
    ],
    "groups": [
        "Adjust {smiles} to {direction} its solubility by about 1 logS, but keep the following groups intact: {pretty_groups}",
        "Modify {smiles} to achieve a {direction} in solubility of approximately 1 logS while preserving these functional groups: {pretty_groups}",
        "How would you alter {smiles} to {direction} its solubility by about 1 logS unit without changing these key groups: {pretty_groups}?",
        "Suggest structural changes to {smiles} that would {direction} its solubility by roughly 1 logS while maintaining these groups: {pretty_groups}",
        "Design a variant of {smiles} with {direction}d solubility (by about 1 logS) that retains all of these intact functional groups: {pretty_groups}",
    ],
}

SMILES_FROM_FORMULA_PROMPTS = [
    "Propose a molecule that has the following formula: {formula}.",
    "Generate a SMILES representation for a compound with the formula {formula}.",
    "What is a plausible SMILES for a compound with the formula {formula}?",
    "Given the formula {formula}, can you suggest a possible SMILES structure?",
    "Create a SMILES representation for a molecule that corresponds to the formula {formula}.",
    "Identify a potential SMILES for a compound with the molecular formula {formula}.",
    "What SMILES structure could correspond to the formula {formula}?",
    "Generate a plausible SMILES for a compound with the formula {formula}.",
    "Given the formula {formula}, what would be a reasonable SMILES representation?",
    "Propose a SMILES structure for a molecule with the formula {formula}.",
    "Generate a SMILES representation for a compound with the formula {formula}.",
]
