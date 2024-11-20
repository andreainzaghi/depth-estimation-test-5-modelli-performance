# Depth Estimation Models Test

Questo progetto contiene diversi modelli di stima della profondità, ciascuno organizzato in cartelle separate, insieme a script specifici per il loro funzionamento, immagini di input per la validazione e un file `requirements.txt` per l'installazione delle dipendenze necessarie.

---

## Contenuto del progetto

### **Modelli di stima della profondità**
1. **Depth-Anything-V2**
   - **Script principale**: `depth_aniting.py`
   - Descrizione: Modello avanzato per la stima della profondità con funzionalità aggiuntive.
   
2. **DPT**
   - **Script principale**: `DPT_run_monodepth.py`
   - Descrizione: Modello basato su DPT (Dense Prediction Transformer) per la stima accurata della profondità.

3. **Marigold**
   - **Script principale**: `mary.py`
   - Descrizione: Modello di nuova generazione per la stima della profondità, ottimizzato per immagini ad alta risoluzione.

4. **ML-Depth-Pro**
   - **Script principale**: `apple_depth_pro.py`
   - Descrizione: Modello ottimizzato per dispositivi Apple e altre piattaforme mobili.

5. **MonoDepth2**
   - **Script principale**: `monodepth2.py`
   - Descrizione: Una delle implementazioni più popolari per la stima monoculare della profondità.

---

### **Dati di input e validazione**
- **Directory**: `img&val`
  - Contiene immagini di test e file per la validazione dei modelli.
  - Descrizione: Include immagini di esempio per verificare la correttezza e l'accuratezza delle mappe di profondità generate.

---

### **Dipendenze e installazione**
- **File**: `requirements.txt`
  - Contiene tutte le librerie e le dipendenze necessarie per eseguire gli script.
  - Comando per installare:  
    ```
    pip install -r requirements.txt
    ```
  - Assicurati di eseguire questo comando in un ambiente virtuale 'venv' Python configurato correttamente.

---