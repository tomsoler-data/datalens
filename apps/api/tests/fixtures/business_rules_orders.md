# Documentation métier synthétique — système de commandes

Ce document décrit les conventions utilisées par le système fictif
de gestion des commandes utilisé pour les tests de DataLens.

Il constitue une fixture de développement.
Il ne représente pas une documentation métier réelle.


## order_id

Le champ `order_id` est l'identifiant unique d'une commande.

Chaque commande enregistrée doit posséder un `order_id`.

Une valeur vide dans `order_id` n'est pas une valeur métier valide.
Elle indique un problème d'import, de collecte ou de génération
de l'identifiant et doit faire l'objet d'une vérification.

Deux commandes différentes ne doivent pas partager le même
`order_id`.


## discount_rate

Le champ `discount_rate` représente le taux de remise appliqué
à une commande.

La valeur est exprimée en pourcentage.

Dans le système source, lorsqu'aucune remise n'est appliquée,
le champ `discount_rate` est volontairement laissé vide.

Une valeur vide de `discount_rate` doit donc être interprétée
comme une remise de 0 %.

Cette convention concerne uniquement `discount_rate` et ne doit
pas être généralisée aux autres variables contenant des valeurs
manquantes.


## customer_segment

Le champ `customer_segment` représente le segment commercial
attribué au client au moment de la commande.

Les exemples de segments existants sont Premium, Standard
et Basic.

Une valeur vide dans `customer_segment` signifie que le client
n'a pas encore été segmenté.

Une valeur manquante dans `customer_segment` ne signifie donc
pas que le client appartient au segment le plus fréquent.

Elle ne doit pas être remplacée automatiquement par Premium,
Standard, Basic ou par le mode statistique de la colonne.

Le statut non segmenté doit être conservé comme une information
distincte tant qu'aucune règle métier supplémentaire n'est fournie.


## age

Le champ `age` représente l'âge déclaré du client en années
au moment de la commande.

L'âge est une variable numérique exprimée en années complètes.

Les valeurs supérieures à 120 dans `age` sont considérées comme
potentiellement invalides par le système métier.

Une valeur de `age` supérieure à 120 doit être vérifiée avant
utilisation analytique.

Une telle valeur ne doit pas être supprimée automatiquement :
elle peut provenir d'une erreur de saisie, d'un problème de source
ou nécessiter une investigation spécifique.


## signup_date

Le champ `signup_date` correspond à la date d'inscription
initiale du client.

Le format attendu dans le système source est ISO `YYYY-MM-DD`.

Une valeur non convertible en date dans `signup_date` doit être
considérée comme un problème de qualité nécessitant une vérification
de la source.

Une date invalide ne doit pas être remplacée automatiquement par
la date du jour ou par une date moyenne.


## shipping_country

Le champ `shipping_country` représente le pays de livraison
de la commande.

Les valeurs doivent correspondre au pays réellement utilisé
pour l'expédition.

Une valeur manquante dans `shipping_country` signifie que
l'information de livraison n'était pas disponible dans l'extraction.

Cette règle ne donne aucune information concernant `discount_rate`,
`customer_segment`, `age` ou `signup_date`.


## order_status

Le champ `order_status` décrit l'état opérationnel d'une commande.

Les valeurs usuelles sont pending, paid, shipped, delivered
et cancelled.

Les catégories de `order_status` ne doivent pas être fusionnées
uniquement parce que leurs fréquences sont faibles.

Toute normalisation de cette colonne doit préserver la signification
opérationnelle des différents états.