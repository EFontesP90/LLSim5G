# LLSim5G

<div style="text-align: justify;">

If you are looking to implement personalized link-level simulations of a heterogeneous 5G network and assess the
performance of your proposed algorithms, this could be a solution, facilitating your upcoming research activities. 
The LLSim5G is an open-source link-level simulator (LLS) developed in Python to recreate heterogeneous 5G use cases.
The rationale behind this proposal is to tailor a controlled simulated environment where it could coexist
terrestrial networks (TNs) and non-terrestrial networks (NTNs) with multiple concurrent users under diverse mobility behaviors
and reception conditions, requesting differentiated broadband traffic from the available infrastructure. We intend to
create a usable and comprehensive solution in which the outcomes can be easily integrated into data-driven and 
machine-learning research approaches for multiple applications.

The implemented link channel models for TN and NTN comply with the 3GPP standards TR 38.901 [1] and TR 38.811 [2],
respectively, for frequencies from 0.5 to 100 GHz. For the NTN use cases, it could be simulated the satellite service links
from spaceborne or airborne platforms to multiple handheld or IoT end devices. Regarding TN, the LLSim5G allows recreating 
all the scenarios defined in TR 38.901, such as Urban Macro (UMa), Urban Micro (UMi), Rural Macro (RMa), Indoor-Hotspot
(InH) and Indoor-Factory (InF). Moreover, we add the capability of simulating unmanned aerial vehicles (UAVs) acting as
5G base stations (BSs) following the Urban Aerial-to-Ground (UAG) channel model as described in [3]. In any case, the 
simulated use case could be a complex three-dimensional (3D) heterogeneous network (HetNet) with multiple TNs, NTNs, and 
UAVs covering the desired service area.

Regarding the end devices (EDs), their configuration for the link computation is compliant with the 3GPP standards 
TR 38.901 [1] and TR 38.811 [2], and subject to the considered network. The resulting link computation could be 
subject to (s.t.), if enabled, fast-fading (FF), shadowing fading (SF), atmospheric absorption (AA), line-of-sight (LOS)
or non-line-of-sight (NLOS), outdoor-to-indoor (O2I) penetration, and human blockage (HB). In the simulation, we can
define multiple (only constrained for the resulting simulation time) EDs freely distributed in the service area, with
either static or complex mobility behaviors (as described below) and different velocities. Among the EDs, we add the capability
of bidirectional link computation through device-to-device (D2D) communication (if enabled) with multiple applications
such as D2D multicasting (D2DM), forwarding device (FD) selection, or vehicle-to-vehicle (V2V) communication.

The physical (PHY) link layer of a wireless communication system is typically simulated through LLSs, mathematically 
modeling the point-to-point communication of a paired transmitter (Tx) and receiver (Rx) with a detailed link characterization
and evaluating metrics such as bit/block error rate (BER/BLER) and signal-to-interference-plus-noise ratio (SINR) (typically
used as reference inputs to system-level simulators (SLS)) [4]. Multiple research fields could take advantage of LLSim5G for
advancing fields such as radio resource management, network slicing, V2V and D2D communications, interference management,
channel estimation, MIMO and Adaptive Modulation and Coding scheme. In such context, the initial version of the
LLSsim5G could be a useful tool for addressing various open research challenges. 

## Table of Contents

- [Main Features](#installation)
- [Structure](##contributing)
- [Installation/Requirements](#contributing)
- [How to use](#contributing)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License and Citation](#acknowledgements)
- [Additional Support](#support)
- [Acknowledgements](#acknowledgements)
- [References](#acknowledgements)


## Main Characteristics

Table I summarizes the main LLSim5G characteristics for this initial version (v.1.0.0).

Table I. LLSim5G main parameters.

| Parameters                         | Value (v.1.0.0)                                                                                                                     |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Scenarios                          | TN: UMa, UMi, RMa, InH, InF, <br/> NTN: HAPS, LEO, MEO, GEO,<br/>UAV: A2G                                                           |
| Link Modes                         | TN/UAV: downlink (DL), uplink (UL), <br/> NTN: DL<br/>  D2D                                                                         |
| Network Topology                   | Single Cell or Multicell                                                                                                            |
| EDs type                           | pedestrian, vehicle, IoT                                                                                                            |
| Frequency range (GHz)              | 0.5-100                                                                                                                             |
| NR numerology                      | 0-4, s.t. 3GPP TS 38.214 [5]                                                                                                        |
| Path Loss models                   | TN/D2D: TR 38.901,<br/> NTN: TR 38.811,<br/>A2G: [3]                                                                                |
| Large scale fading models          | TN/D2D/UAV: TR 38.901,<br/> NTN: TR 38.811                                                                                          |
| Fast fading models                 | TN/D2D/UAV: TDL, CDL (s.t. TR 38.901), Jakes (s.t. [6]),<br/> NTN: TDL (s.t. TR 38.811), Jakes (s.t. [6])                           |
| Propagation conditions             | O2O (LOS/NLOS), O2I (NLOS), I2I (LOS/NLOS)                                                                                          |
| Link to system Adaptation          | BLER and CQI estimation                                                                                                             |
| Atmospheric Absorption             | yes, TN/A2G: TR 38.901,<br/> NTN: TR 38.811                                                                                         |
| Antenna Model                      | TN/UAV/D2D: s.t. TR 38.901 (Omnidirectional, Three sectors (120°), Four sectors (90°))  <br/> NTN: s.t. TR 38.811                   |
| Antenna Polarization               | single, dual                                                                                                                        |
| EDs mobility models                | Stationary,  Linear, Random Walk, Random Waypoint,<br/>Random Direction, Truncated Levy Walk, Gauss-Markov, Gauss-Markov (s.t. [7]) |            
| NTN and UAV mobility               | Not available in this current version                                                                                               |
| MIMO                               | Not available in this current version                                                                                               |
| Interference considerations        | Not available in this current version                                                                                               |
| [Outputs](./SimulationOutputs.pdf) | BLER, SINR, CQI, nodes coordinates, and nodes movement video                                                                        |             

## Structure

Our open-source LLS is wholly programmed in Python language with a modular structure and flow process, according to
Fig. 1. The first step is the initialization of the parameters and simulation settings, such as simulation time,
resolution, grid size, type and the number of users, mobility models, the number of TNs, NTNs, and UAVs covering the desired service
area, among multiple others, as defined in [SimulationConfiguration](./SimulationConfiguration.pdf). Once the 
simulator’s initialization and resource pool selection is finalized, it starts the iterative link computation
along the defined simulation time. The links are computed among all the available BSs and EDs and subject to the selected
link modes, s.t. Table I. In the case of D2D, the simulator calculates the D2D link among all the EDs (with the D2D mode
enabled) independently of their distance. The link computation ends with the resulting SINR for each enabled link. Then,
the link to system abstraction is executed, where each SINR value is estimated for the BLER and the corresponding CQI.



<img src="general/img/SchemeFlow.png" alt="Diagram" width="500">

Fig. 1: LLSim5G abstraction model. 

[//]: # (![LLSim5G abstraction model]&#40;general/img/SchemeFlow.png&#41;)
[//]: # (Fig. 1: LLSim5G abstraction model )

Fig. 2 presents the internal structure of the simulator, including the dependencies of the .py files, as well as the inputs and 
outputs. The details about the overall outputs of the simulator can be found in [SimulationOutputs](./SimulationOutputs.pdf).

<img src="general/img/LLSim5g_IStructure.png" alt="Diagram" width="700">

Fig. 2: LLSim5G internal structure.

[//]: # (![LLSim5G internal structure]&#40;general/img/LLSim5g_IStructure.png&#41;)
[//]: # (Fig. 1: LLSim5G internal structure)

## Installation and Requirements

You need to clone the project from GitHub using the link: https://github.com/EFontesP90/lls_tn-ntn_5g.git;
or download the project ZIP file.
Then, you need to check that all the requirements, main dependencies, and external libraries and versions are satisfied 
([requirements](./requirements.txt)).

## How to use

First, you need to carefully check the [SimulationConfiguration](./SimulationConfiguration.pdf) and 
[SimulationOutputs](./SimulationOutputs.pdf) files and the additional complementary documents. 
Further, we include the file [Scenario_Description_example1](./Scenario_Description_example1.pdf), describing
a possible use case with the corresponding initialization setup and application.

The LLSim5G has been the baseline platform for the assessments of the contributions proposed in:

- E. F. Pupo, C. C. González and M. Murroni, "Multi-rate Multicasting Over Fixed Pre-Computed MIMO Beams," 2024 IEEE 
International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB), Toronto, ON, Canada, 2024, pp. 1-6, 
doi: 10.1109/BMSB62888.2024.10608199.
- A. Collu, E. F. Pupo, C. C. González, and M. Murroni, "Exploiting the benefits of in-band D2D communications for 
5G-MBS use cases," 2024 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB), 
Toronto, ON, Canada, 2024, pp. 1-6, doi: 10.1109/BMSB62888.2024.10608283.
- C. C. González, E. F. Pupo, J. Montalban, E. Iradier, P. Angueira and M. Murroni, "Federated Learning-based 
Unicast/Multicast Service Delivery over 6G O-RAN Framework," 2024 IEEE International Symposium on Broadband Multimedia
Systems and Broadcasting (BMSB), Toronto, ON, Canada, 2024, pp. 1-6, doi: 10.1109/BMSB62888.2024.10608261.
- C. C. González et al., "A QoE-based Energy-aware Resource Allocation Solution for 5G Heterogeneous Networks," 
2024 16th International Conference on Quality of Multimedia Experience (QoMEX), Karlshamn, Sweden, 2024, pp. 29-35, 
doi: 10.1109/QoMEX61742.2024.10598282.
- C. C. González, E. F. Pupo, E. Iradier, P. Angueira, M. Murroni and J. Montalban, "Network Selection Over 5G-Advanced
Heterogeneous Networks Based on Federated Learning and Cooperative Game Theory," in IEEE Transactions on Vehicular 
Technology, vol. 73, no. 8, pp. 11862-11877, Aug. 2024, doi: 10.1109/TVT.2024.3373638.
- E. Fontes Pupo, C. Carballo González, J. Montalban, P. Angueira, M. Murroni and E. Iradier, "Artificial Intelligence 
Aided Low Complexity RRM Algorithms for 5G-MBS," in IEEE Transactions on Broadcasting, vol. 70, no. 1, pp. 110-122, 
March 2024, doi: 10.1109/TBC.2023.3311337.
- E. F. Pupo, C. C. Gonzalez, V. Popescu, D. Giusto, and M. Murroni, "Beyond 5G Multicast for XR Communications aided
by Pre-computed Multi-beams and NOMA," 2023 IEEE Globecom Workshops (GC Wkshps), Kuala Lumpur, Malaysia, 2023, 
pp. 738-743, doi: 10.1109/GCWkshps58843.2023.10465216.
- C. C. González, E. F. Pupo, J. Montalban, S. Pizzi, E. Iradier and M. Murroni, "Hybrid Terrestrial-Airborne
Connectivity for Unicast/Broadcast Services Beyond 5G," 2023 IEEE International Symposium on Broadband Multimedia 
Systems and Broadcasting (BMSB), Beijing, China, 2023, pp. 1-6, doi: 10.1109/BMSB58369.2023.10211608.

This software is provided on an "as is" basis, without warranties of any kind, either express or implied, including,
but not limited to, warranties of accuracy, adequacy, validity, reliability, or compliance for any specific purpose.
Neither the University of Cagliari nor the authors of this software are liable for any loss, expense, or damage of
any type that may arise in using this software.

## Limitations

The current version of LLSim5G v.1.0.0 is subject to the following limitations (some of them are underdevelopment
capabilities):

- No support for MIMO.
- No support for mobility simulations on the BSs (i.e., the defined UAVs or NTNs) side.
- The fast-fading CDL model is not enabled for NTN simulations.
- No interference considerations.
- Lack of Parallelization and GPU Support. 

Of course, the identified limitations and missing capabilities could be infinity regarding personal research interests or
other available simulators. Nevertheless, we are highlighting the current underdevelopment works, which we
identify as important for the rationale behind the project. In any case, we encourage any interested researcher to help
us advance this simulator toward a more comprehensive and usable solution. 

## Contributing

Ernesto Fontes Pupo, orcid: "0000-0002-1715-6015", e-mail: efontesp90@gmail.com, Google Scholar:
https://scholar.google.com/citations?user=2-FLQJEAAAAJ&hl=en.

Claudia Carballo Gonzalez, orcid: "0000-0002-6429-1375", e-mail: ccgclaudia7892@gmail.com, Google Scholar:
https://scholar.google.com/citations?user=E0oft9oAAAAJ&hl=en&oi=ao.

The LLS was developed during their Ph.D. studies with the Department of Electrical and Electronic Engineering
(DIEE/UdR CNIT), University of Cagliari.
https://web.unica.it/unica/en/homepage.page

## License and Citation

                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

The LLSim5G is Licensed under the GNU General Public License v3.0 ([LICENSE](/LICENSE));
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

https://www.gnu.org/licenses/gpl-3.0.en.html

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. Please look at the License for the 
specific language governing permissions and limitations under the License.

    LLSim5G is a link-level simulator for HetNet 5G use cases.
    Copyright (C) 2024  Ernesto Fontes, Claudia Carballo

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

If you use this software or part of it for your research, please cite the work [8]:

```bibtex
@INPROCEEDINGS{10211507,
  author={Pupo, Ernesto Fontes and González, Claudia Carballo and Iradier, Eneko and Montalban, Jon and Murroni, Maurizio},
  booktitle={2023 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB)}, 
  title={5G Link-Level Simulator for Multicast/Broadcast Services}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/BMSB58369.2023.10211507}}
```

Plain text:

E. F. Pupo, C. C. González, E. Iradier, J. Montalban and M. Murroni, "5G Link-Level Simulator for Multicast/Broadcast
Services," 2023 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB), Beijing, China,
2023, pp. 1-6, doi: 10.1109/BMSB58369.2023.10211507.

## Additional Support

Please contact the contributing authors, e-mail: efontesp90@gmail.com, ccgclaudia7892@gmail.com. 


## Acknowledge

Thanks to the Department of Communications Engineering, University of the Basque Country (UPV/EHU), where the authors
stayed during their Ph.D. period abroad.

## References

[1] G. T. 38.901, “5G; Study on channel model for frequencies from 0.5 to 100 GHz (3GPP TR 38.901 version 16.1.0 Release 16),” 2020.

[2] TR 38.811, “Technical Specification Group Radio Access Network; Study on New Radio (NR) to support non-terrestrial networks (Release 15),” 2020.

[3] W. Khawaja, I. Guvenc, D. W. Matolak, U.-C. Fiebig, and N. Schneck enburger, “A survey of air-to-ground propagation channel modeling for
unmanned aerial vehicles,” IEEE Communications Surveys & Tutorials, vol. 21, no. 3, pp. 2361–2391, 2019.

[4] M. Rupp, S. Schwarz, and M. Taranetz, “The vienna lte-advanced simulators,”
https://link. springer. com/book/10.1007% 2F978-981-10 0617-3, 2016.

[5] “5g; NR; physical layer procedures for data,” 3GPP, Sophia Antipolis, France, 3GPP Rep. TS 38.214 version 16.2.0
release 16, 2020.

[6] P. Dent, G. E. Bottomley, and T. Croft, “Jakes fading model revisited,” Electronics letters, vol. 13, no. 29,
pp. 1162–1163, 1993.

[7] https://github.com/panisson/pymobility

[8] E. F. Pupo, C. C. González, E. Iradier, J. Montalban and M. Murroni, "5G Link-Level Simulator for Multicast/Broadcast
Services," 2023 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB), Beijing, China,
2023, pp. 1-6, doi: 10.1109/BMSB58369.2023.10211507.

</div>
