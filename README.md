## CHL-DTI:
A novel high-low order information convergence framework for effective drug-target interaction prediction
### Quick start
We provide an example script to run experiments on our dataset: 

- Run `./CHL-DTI/code/main.py`: predict drug-target interactions. 

### All process
 -Run `./code/main.py`   You can run the entire model


### Code and data

#### 
- `model.py`: CHL model
- `GCN.py`: a GCL layers
- `loaddata_utils.py`: used for loading data
- `main.py`: use the dataset to run CHL-DTI 
- `hypergraph_utils.py`: Hypergraph related toolkits
- `Structe_DPP_HyperGraph.py`: strategy to build hypergraph for DPP

#### data sample `data/Luo` directory
- `drug.txt`: list of drug names
- `protein.txt`: list of protein names
- `disease.txt`: list of disease names
- `se.txt`: list of side effect names
- `drug_dict_map`: a complete ID mapping between drug names and DrugBank ID
- `protein_dict_map`: a complete ID mapping between protein names and UniProt ID
- `mat_drug_se.txt` 		: Drug-SideEffect association matrix
- `mat_protein_protein.txt` : Protein-Protein interaction matrix
- `mat_protein_drug.txt` 	: Protein-Drug interaction matrix
- `mat_drug_protein.txt`: Drug_Protein interaction matrix
- `mat_drug_drug.txt` 		: Drug-Drug interaction matrix
- `mat_protein_disease.txt` : Protein-Disease association matrix
- `mat_drug_disease.txt` 	: Drug-Disease association matrix
- `Similarity_Matrix_Drugs.txt` 	: Drug similarity scores based on chemical structures of drugs
- `Similarity_Matrix_Proteins.txt` 	: Protein similarity scores based on primary sequences of proteins



