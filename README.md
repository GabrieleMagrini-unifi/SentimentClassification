# SentimentClassification
AI project
Il progetto è fromato da 5 file .py, ognuno adibito ad uno scopo preciso.<br/>
Per riprodurre i risultati conseguiti nella relazione, o più in generale per far eseguire correttaemnte il codice, è sufficiente seguire pochi semplici step.<br/>
<br/>
Innanzitutto, preparare il dataset.<br/>
<br/>
Se abbiamo un file .tsv già pronto, basterà caricarlo attraverso il comando "pd.read_csv()", presente nella 11esima riga di tutti i file.<br/>
Altrimenit, possiamo usare "DatasetCreator", nel caso in cui il dataset sia in forma testuale e spezzettato in più recensioni.<br/>
In tal caso, vanno rinominate in modo tale da riprodurre il nome "pos (i)" per le positive(con i =1,...,numero di review positive) <br/>
e "neg (i)" per le negative(con i =1,...,numero di review negative).<br/>
<br/>
A questo punto,basta linkare la cartella contenenti le review nell'ottava e quindicesima riga di "DataframeCreator".<br/>
Il risultato sarà un file .tsv, che alterna in ogni riga una review positiva e una negativa.(se il numero di review positive e negative è identico).<br/>
Dopodichè,il programma è quasi completamente automatico: basterà infatti inserire il file .tsv nella 11esima riga del file desiderato,nel metodo<br/>
"pd.read_csv()" e come risultato sarà stampato a schermo l'accuratezza dei due classificatori(Perceptron e Decision tree) dato il particolare dataset.<br/>
<br/>
Per comodità, sono posti nella cartella "data" diversi dataset pronti, dove i file di tipo "data_review_" sono quelli referenti agli esperimenti di Pang e Lee,<br/>
mentre "LabeledDataTraining.csv" si riferisce al dataset di 25.000 recensioni presi da IMDB e già ordinato e etichettato.<br/>
Quest'ultimo dataset è reperibile liberamente all'indirizzo "https://www.kaggle.com/c/word2vec-nlp-tutorial/data".<br/>
Infine sono presenti 3 recensioni testuali, 1 fittizia e scritta da me, positiva ma con riserva, e altre due(anch'esse positive) riadattate da altrettante recensioni<br/>
di popolari videogiochi(prese da www.ign.com/articles).<br/>
<br/>
Queste ultime recensioni servono a testare l'accuratezza ei classificatori in particolari casi, ovvero dove la recensione sia discordante col sentimento finale
o dove sia presente un numero elevato di aggettivi altisonanti("non è buono tanto quanto","non è cattivo ma",ecc).<br/>
L'unico file dove sono state testate tuttavia è nin "SentimentAnalysis.py": se si vuole riprodurre un simile risultato con un proprio file .txt, basterà inserirlo
nella riga 15, al posto di "data/receFIFA21.txt".<br/>
Gli algoritmi usati nel programma per la trasformazione del dataset(tagging, ecc) sono stati tutti scritti da me, mentre la funzione "simple_split" è stata presa dal <br/>
video "Hands-on Scikit-learn for Machine Learning: Bag-of-Words Model and Sentiment ", reperibile all'indirizzo "https://www.youtube.com/watch?v=KE53PAfVJ5c",<br/>
che nel complesso mi ha aiutato nella stesura di una prima versione del programma.<br/>
<br/>
Infine, ecco come simulare i risultati da me ottenuti in precedenza:<br/>
<br/>
Una volta selezionato il dataset"data_review_balanced", basterà eseguire i 4 file in successione, ognuno dei quali stampeà il risultato per quella particolare
tecnica usata("SentimentAnalysis.py" si basa sui semplici unigrammi, "SentimentAnalysisPOS.py" sfrutta anche il POS tagging, "SentimentAnalysisAdjectives.py" filtra
le sole word "aggettivo" dal POS tagging, "SentimentAnalysisPosition.py" etichetta le parole a seconda della loro posizione dìnel dataset).<br/>
Per simulare infine l'uso di bigrammi o unigrammi+bigrammi, basterà aggiungere ngram_range all'interno di CountVectorizer() alla riga 40 di "SentimentAnalysis.py".<br/>

