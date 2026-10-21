# Third-party code and acknowledgements

The `src/pypret/` package is the exact vendored pypret implementation used by
the report (version string `0.1alpha`), by Nils C. Geib. Its MIT license is
preserved in `src/pypret/LICENSE` and `third_party/pypret/LICENSE`. Source:
https://github.com/ncgeib/pypret. We bundle it so published retrieval behavior
does not depend on a moving upstream version or an external checkout.

COPRA reference: N. C. Geib et al., “Common pulse retrieval algorithm: a fast
and universal method to retrieve ultrashort pulses,” Optica 6, 495–505 (2019).

The COPRA/Hann/PnP-COPRA utilities and pulse catalogue were developed from the
project's earlier work with Abhimanyu Borthakur. The COPRA Flow report is by
Atal Narayan Sahu, in collaboration with Abhimanyu Borthakur and Sergio
Carbajo, The Quantum Light-Matter Cooperative, UCLA.

Experimental TG-FROG measurements were supplied by Linshan. They are not
included in this code distribution. Their redistribution and data license
must be agreed separately. The source-code license does not license those
measurements, model weights, or training-data downloads.
