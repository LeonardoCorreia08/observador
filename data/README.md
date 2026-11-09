# Dados e Datasets

Os datasets **não são incluídos neste repositório** devido ao tamanho dos arquivos e diretrizes de licenciamento. 

**Regras gerais para uso de dados neste projeto:**
* Registe a origem, licença, versão, data de obtenção, número de imagens por split e procedimento de anotação.
* Não publique dados privados, imagens sem autorização ou dados pessoais. 
* Se os dados forem extraídos de vídeos ou outras fontes (ex: web scraping), documente detalhadamente o processo de amostragem e filtragem, garantindo os direitos autorais e de privacidade.

Abaixo está o registro (`Data Card`) dos datasets utilizados no treinamento, validação e testes dos experimentos do projeto **Observador**.

---

## 1. COCO (Common Objects in Context)

Utilizado primariamente para tarefas de detecção de objetos estáticos e validação geral do modelo (ex: `exp_001` e `exp_002`).

* **Origem:** [COCO Consortium](https://cocodataset.org/)
* **Licença:** Creative Commons Attribution 4.0 License (CC BY 4.0)
* **Versão:** COCO 2017
* **Data de obtenção:** Novembro de 2026 *(Atualize conforme a data do seu download)*
* **Splits (Imagens):**
  * **Train:** ~118.287 imagens
  * **Val:** 5.000 imagens
  * **Test:** ~40.670 imagens
* **Classes:** 80 categorias de objetos (pessoas, veículos, animais, objetos cotidianos, etc).
* **Procedimento de Anotação:** Anotação crowdsourced via Amazon Mechanical Turk. Imagens extraídas do Flickr, anotadas com bounding boxes, máscaras de segmentação e keypoints, passando por rigorosa verificação de qualidade em múltiplas etapas.

---

## 2. MOT17 (Multiple Object Tracking 2017)

Utilizado para avaliar e testar os algoritmos de rastreamento de múltiplos objetos (Tracking), como o ByteTrack avaliado no `exp_003` (sequência `MOT17-02-DPM`).

* **Origem:** [MOTChallenge](https://motchallenge.net/data/MOT17/)
* **Licença:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0) - *Restrito para uso acadêmico/não-comercial*.
* **Versão:** MOT17
* **Data de obtenção:** Novembro de 2026 *(Atualize conforme a data do seu download)*
* **Splits (Sequências de Vídeo):**
  * **Train:** 7 sequências (com anotações de Ground Truth para rastreamento completo)
  * **Test:** 7 sequências (anotações ocultas, submetidas ao servidor de avaliação)
* **Classes:** Focado principalmente em pedestres (pedestrians) em ambientes estáticos e em movimento.
* **Procedimento de Anotação:** Bounding boxes altamente precisas anotadas frame a frame. Cada objeto rastreável recebe um `track_id` único que persiste ao longo da sequência, além de anotações sobre oclusão e visibilidade do objeto em cada frame.

---

## 3. BDD100K (Berkeley DeepDrive)

Utilizado para expandir a capacidade do modelo em contextos de direção autônoma, variações climáticas, detecção veicular avançada e cenários noturnos/chuvosos.

* **Origem:** [UC Berkeley / BDD](https://bdd-data.berkeley.edu/)
* **Licença:** BSD 3-Clause License (para uso educacional e de pesquisa).
* **Versão:** BDD100K (Images/Detection & Tracking)
* **Data de obtenção:** Novembro de 2026 *(Atualize conforme a data do seu download)*
* **Splits (Imagens para Detecção):**
  * **Train:** 70.000 imagens
  * **Val:** 10.000 imagens
  * **Test:** 20.000 imagens
* **Classes:** 10 categorias focadas em trânsito (pedestre, ciclista, carro, caminhão, ônibus, trem, motocicleta, bicicleta, semáforo, placa de trânsito).
* **Procedimento de Anotação:** Extraído de milhares de horas de vídeos de direção em diferentes horários do dia e condições climáticas nos EUA. As anotações de bounding box são feitas por anotadores profissionais seguindo diretrizes rígidas, capturando também atributos meta (clima, hora do dia, cena da via).