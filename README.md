# LLSim5G

If you are looking for implementing personalized link level simulations of a heterogeneous 5G network and assess the
performance of your proposed algorithms, this could be a solution, facilitating your upcoming research activities. 
The LLSim5G is an open-source link-level simulator (LLS) developed in Python to recreate heterogeneous 5G use cases.
The rationale behind this proposal is being able to tailor a controlled simulated environment where it could coexist
terrestrial (TN) and non-terrestrial networks (NTN) with multiple concurrent users under diverse mobility behaviors,
reception conditions, and requesting differentiated broadband traffic from the available infrastructure.

The link channel models implemented for TN and NTN are compliant with the 3GPP standards TR 38.901 and TR 38.811
respectively.

The physical (PHY) link layer of a wireless communication systems is typically simulated through LLSs, mathematically 
modeling the point-to-point communication of a pair transmitter (Tx) and receiver (Rx) with a detailed link characterization
and evaluating metrics such as bit/block error rate (BER/BLER) and signal-to-interference-plus-noise ratio (SINR) (typically
used as reference inputs to system-level simulators (SLS)) [6]. Multiple research fields take advantage of LLSs for
advancing filed such as radio resource management, interference management, channel estimation, Multiple-Input
Multiple-Output (MIMO), and Adaptive Modulation and Coding (AMC) scheme. 

It includes channel modeling, modulation, coding schemes, and performance 
analysis features for research and
educational applications in wireless communication. In compliance with the 3GPP standards (TR 38.901 and TR 38.811),
the LLS allows the modeling of terrestrial and non-terrestrial networks (TNs-NTNs), multiple user types distributed in
the defined scenarios, and different mobility behaviors.

This software is provided on an "as is" basis, without warranties of any kind, express or implied, including, but not
limited to, warranties of accuracy, adequacy, validity, reliability, or compliance for any specific purpose. Neither the
University of Cagliari nor the authors of this software are liable for any loss, expense, or damage of any type that may
arise in using this software.

If you include this software or part of it within your own software, README and LICENSE files cannot be removed from it
and must be included in the root directory of your software package.  Moreover, if you use this 5G LLS in your research,
please cite it as follows:

Pupo, Ernesto Fontes, Claudia Carballo González, Eneko Iradier, Jon Montalban, and Maurizio Murroni. "5G link-level
simulator for multicast/broadcast services." In 2023 IEEE International Symposium on Broadband Multimedia Systems and
Broadcasting (BMSB), pp. 1-6. IEEE, 2023, doi: 10.1109/BMSB58369.2023.10211507.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Core Contributors](#contributors)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#acknowledgements)


## Installation

Instructions on how to install your project. For example:

```bash
pip install your_project_name
```

## Instructions on how to use your project.

```bash
import your_project_name
```

# Example usage

```bash
your_project_name.some_function()
```

# Features
List of features your project provides.

# Core Contributors

Ernesto Fontes Pupo, orcid: "0000-0002-1715-6015", affiliation: "University of Cagliari", e-mail: efontesp90@gmail.com

Claudia Carballo Gonzalez, orcid: "0000-0002-6429-1375", affiliation: "University of Cagliari", e-mail: ccgclaudia7892@gmail.com

# Contributing

Guidelines on how to contribute to your project. For example:

-Fork the repository
-Create a new branch (git checkout -b feature/feature-name)
-Commit your changes (git commit -m 'Add some feature')
-Push to the branch (git push origin feature/feature-name)
-Open a Pull Request

## License

This project is licensed under the Apache License 2.0. For details, see the [LICENSE](./LICENSE) file.

## Acknowledgements

- Thanks to [Person](https://github.com/person) for their guidance.
- [Library](https://link-to-library) for providing useful resources.

---

*This project is a part of XYZ initiative and is maintained by [Your Name](https://github.com/your-github-profile).*

For any questions or issues, please contact [Your Email](mailto:your-email@example.com).

# Acknowledgements

Credits and acknowledgements for resources, libraries, inspiration, etc.

## References

G. Nardini, D. Sabella, G. Stea, P. Thakkar, A. Virdis, "Simu5G – An OMNeT++ Library for End-to-End Performance
Evaluation of 5G Networks," in IEEE Access, vol. 8, pp. 181176-181191, 2020, doi: 10.1109/ACCESS.2020.3028550.

[6] M. Rupp, S. Schwarz, and M. Taranetz, “The vienna lte-advanced simulators,” https://link. springer. com/book/10.1007% 2F978-981-10
0617-3, 2016.

[14] W. Khawaja, I. Guvenc, D. W. Matolak, U.-C. Fiebig, and N. Schneck enburger, “A survey of air-to-ground propagation channel modeling for
unmanned aerial vehicles,” IEEE Communications Surveys & Tutorials, vol. 21, no. 3, pp. 2361–2391, 2019.