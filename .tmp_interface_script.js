
        const fieldConfig = [
            { key: 'Age', label: 'Age', type: 'stepper', min: 10, max: 80, step: 1, defaultValue: 21 },
            { key: 'Sex', label: 'Genre', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'Femme' },
                { value: 1, text: 'Homme' }
            ] },
            { key: 'Height', label: 'Taille (cm)', type: 'stepper', min: 130, max: 210, step: 1, defaultValue: 170 },
            { key: 'Overweight_Obese_Family', label: 'Antecedents familiaux de surpoids', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'yes' },
                { value: 1, text: 'no' }
            ] },
            { key: 'Consumption_of_Fast_Food', label: "Consommation frequente d'aliments caloriques", type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'yes' },
                { value: 1, text: 'no' }
            ] },
            { key: 'Frequency_of_Consuming_Vegetables', label: 'Frequence de consommation de legumes (1-3)', type: 'range', min: 1, max: 3, step: 1, defaultValue: 3 },
            { key: 'Number_of_Main_Meals_Daily', label: 'Nombre de repas principaux (1-4)', type: 'range', min: 1, max: 4, step: 1, defaultValue: 2 },
            { key: 'Food_Intake_Between_Meals', label: 'Consommation entre les repas', type: 'select', defaultValue: 2, options: [
                { value: 1, text: 'No' },
                { value: 2, text: 'Sometimes' },
                { value: 3, text: 'Frequently' },
                { value: 4, text: 'Always' }
            ] },
            { key: 'Smoking', label: 'Fumeur', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'no' },
                { value: 1, text: 'yes' }
            ] },
            { key: 'Liquid_Intake_Daily', label: "Consommation d'eau par jour (1-3)", type: 'range', min: 1, max: 3, step: 1, defaultValue: 2 },
            { key: 'Calculation_of_Calorie_Intake', label: 'Suivi calorique', type: 'select', defaultValue: 2, options: [
                { value: 2, text: 'no' },
                { value: 1, text: 'yes' }
            ] },
            { key: 'Physical_Excercise', label: 'Activite physique par semaine (1-4)', type: 'range', min: 1, max: 4, step: 1, defaultValue: 3 },
            { key: 'Schedule_Dedicated_to_Technology', label: "Temps d'utilisation d'appareils electroniques (1-5)", type: 'range', min: 1, max: 5, step: 1, defaultValue: 3 },
            { key: 'Type_of_Transportation_Used', label: 'Moyen de transport', type: 'select', defaultValue: 4, options: [
                { value: 1, text: 'Walking' },
                { value: 2, text: 'Bike' },
                { value: 3, text: 'Motorbike' },
                { value: 4, text: 'Public_Transportation' },
                { value: 5, text: 'Car' }
            ] }
        ];

        const defaultValues = Object.fromEntries(fieldConfig.map((f) => [f.key, f.defaultValue]));

        const fieldsContainer = document.getElementById('fields');
        const orderedKeys = fieldConfig.map((f) => f.key);

        function createRangeField(config) {
            const label = document.createElement('label');
            const wrap = document.createElement('div');
            wrap.className = 'range-wrap';
            const head = document.createElement('div');
            head.className = 'range-head';
            head.innerHTML = `<span>${config.label}</span><strong id="${config.key}_value">${config.defaultValue}</strong>`;
            const input = document.createElement('input');
            input.type = 'range';
            input.id = config.key;
            input.min = config.min;
            input.max = config.max;
            input.step = config.step || 1;
            input.value = config.defaultValue;
            input.oninput = () => {
                const out = document.getElementById(`${config.key}_value`);
                if (out) out.textContent = input.value;
            };
            wrap.appendChild(head);
            wrap.appendChild(input);
            label.appendChild(wrap);
            return label;
        }

        function createSelectField(config) {
            const label = document.createElement('label');
            label.innerHTML = `<span>${config.label}</span>`;
            const select = document.createElement('select');
            select.id = config.key;
            config.options.forEach((opt) => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.text;
                if (Number(opt.value) === Number(config.defaultValue)) option.selected = true;
                select.appendChild(option);
            });
            label.appendChild(select);
            return label;
        }

        function createStepperField(config) {
            const label = document.createElement('label');
            label.innerHTML = `<span>${config.label}</span>`;
            const wrapper = document.createElement('div');
            wrapper.className = 'stepper';
            const input = document.createElement('input');
            input.type = 'number';
            input.id = config.key;
            input.min = config.min;
            input.max = config.max;
            input.step = config.step || 1;
            input.value = config.defaultValue;
            const buttons = document.createElement('div');
            buttons.className = 'buttons';
            const minus = document.createElement('button');
            minus.type = 'button';
            minus.className = 'mini';
            minus.textContent = '-';
            minus.onclick = () => {
                const next = Math.max(Number(config.min), Number(input.value) - Number(config.step || 1));
                input.value = String(next);
            };
            const plus = document.createElement('button');
            plus.type = 'button';
            plus.className = 'mini';
            plus.textContent = '+';
            plus.onclick = () => {
                const next = Math.min(Number(config.max), Number(input.value) + Number(config.step || 1));
                input.value = String(next);
            };
            buttons.appendChild(minus);
            buttons.appendChild(plus);
            wrapper.appendChild(input);
            wrapper.appendChild(buttons);
            label.appendChild(wrapper);
            return label;
        }

        function renderFields() {
            fieldsContainer.innerHTML = '';
            fieldConfig.forEach((config) => {
                let node;
                if (config.type === 'range') node = createRangeField(config);
                else if (config.type === 'select') node = createSelectField(config);
                else node = createStepperField(config);
                fieldsContainer.appendChild(node);
            });
        }

        function fillSample() {
            Object.entries(defaultValues).forEach(([key, value]) => {
                const input = document.getElementById(key);
                if (input) input.value = value;
            });
            document.getElementById('resultBox').innerHTML = '<div class="big">Sample loaded</div><div class="meta">You can submit the sample or edit any field.</div>';
        }

        function clearFields() {
            orderedKeys.forEach((key) => {
                const input = document.getElementById(key);
                if (input) input.value = defaultValues[key];
                const out = document.getElementById(`${key}_value`);
                if (out) out.textContent = String(defaultValues[key]);
            });
            document.getElementById('resultBox').innerHTML = '<div class="big">Valeurs reinitialisees</div><div class="meta">Les valeurs de demonstration ont ete rechargees.</div>';
        }

        async function refreshStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();

                document.getElementById('apiBadge').textContent = data.api_status;
                document.getElementById('mlflowBadge').textContent = data.mlflow_status;
                document.getElementById('latestRun').textContent = data.latest_run_id || 'none';
                document.getElementById('latestRunStatus').textContent = data.latest_run_status ? ('status: ' + data.latest_run_status) : 'Waiting for data...';
                document.getElementById('runs').textContent = data.runs;
                document.getElementById('metrics').textContent = data.metrics;
                document.getElementById('datasets').textContent = data.datasets;
                document.getElementById('inputs').textContent = data.inputs;
                document.getElementById('evaluationDatasets').textContent = data.evaluation_datasets;
                document.getElementById('traceCount').textContent = data.traces;

                document.getElementById('apiBadge').className = data.api_status === 'up' ? 'value good' : 'value bad';
                document.getElementById('mlflowBadge').className = data.mlflow_status === 'up' ? 'value good' : 'value bad';
            } catch (error) {
                document.getElementById('resultBox').innerHTML = `<div class="big bad">Stats unavailable</div><div class="meta">${error}</div>`;
            }
        }

        async function predict() {
            const payload = {};
            for (const key of orderedKeys) {
                const value = document.getElementById(key).value;
                payload[key] = Number(value);
            }

            const resultBox = document.getElementById('resultBox');
            resultBox.innerHTML = '<div class="big">Prediction en cours...</div><div class="meta">Envoi des donnees au modele.</div>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();

                if (!response.ok) {
                    resultBox.innerHTML = `<div class="big bad">Requete invalide</div><div class="meta">${JSON.stringify(data)}</div>`;
                    return;
                }

                const confidence = Number(data.confidence || 0) * 100;
                resultBox.innerHTML = `
                    <div class="big good">${data.predicted_label}</div>
                    <div class="meta">Classe numerique: ${data.predicted_class}</div>
                    <div class="meta">Confiance du modele: ${confidence.toFixed(1)}%</div>
                `;
                refreshStats();
            } catch (error) {
                resultBox.innerHTML = `<div class="big bad">Erreur reseau</div><div class="meta">${error}</div>`;
            }
        }

        renderFields();
        refreshStats();
        setInterval(refreshStats, 5000);
    
