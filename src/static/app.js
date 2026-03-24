document.addEventListener('DOMContentLoaded', () => {
    let currentMode = 'regular';

    const modeBtns = document.querySelectorAll('.mode-btn');
    const form = document.getElementById('generate-form');
    
    const distInput = document.getElementById('distance');
    const tolInput = document.getElementById('tolerance');
    const distVal = document.getElementById('dist-val');
    const tolVal = document.getElementById('tol-val');

    const hillyGroup = document.getElementById('hilly-options');
    const historicGroup = document.getElementById('historic-options');
    const novelGroup = document.getElementById('novel-options');

    const btn = document.getElementById('generate-btn');
    const btnText = document.querySelector('.btn-text');
    const btnSpinner = document.getElementById('btn-spinner');

    const idleState = document.getElementById('idle-state');
    const loadingState = document.getElementById('loading-state');
    const mapFrame = document.getElementById('map-frame');

    // Input Sync
    distInput.addEventListener('input', e => distVal.textContent = e.target.value);
    tolInput.addEventListener('input', e => tolVal.textContent = Math.round(e.target.value * 100) + '%');
    document.getElementById('hilly_factor').addEventListener('input', e => document.getElementById('hilly-val').textContent = e.target.value);
    document.getElementById('downtown_radius').addEventListener('input', e => document.getElementById('radius-val').textContent = e.target.value);
    document.getElementById('novelty_factor').addEventListener('input', e => document.getElementById('novel-val').textContent = e.target.value);

    // Mode Selection Logic
    modeBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            modeBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentMode = e.target.dataset.type;
            
            // Toggle contextual groups
            hillyGroup.classList.add('hidden');
            historicGroup.classList.add('hidden');
            novelGroup.classList.add('hidden');

            if (currentMode === 'hilly') hillyGroup.classList.remove('hidden');
            if (currentMode === 'historic') {
                hillyGroup.classList.remove('hidden');
                historicGroup.classList.remove('hidden');
            }
            if (currentMode === 'novel') novelGroup.classList.remove('hidden');
        });
    });

    // API Trigger
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            type: currentMode,
            distance: parseFloat(distInput.value),
            tolerance: parseFloat(tolInput.value),
            hilly_factor: parseFloat(document.getElementById('hilly_factor').value),
            downtown_radius: parseFloat(document.getElementById('downtown_radius').value),
            novelty_factor: parseFloat(document.getElementById('novelty_factor').value),
        };

        // Transition to Loading
        btn.disabled = true;
        btnText.textContent = 'Generating...';
        btnSpinner.classList.remove('hidden');

        idleState.classList.add('hidden');
        mapFrame.classList.add('hidden');
        loadingState.classList.remove('hidden');
        mapFrame.src = "";

        try {
            const res = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();

            if (data.status === 'success') {
                // Success: display Folium map
                mapFrame.src = data.map_url + "?t=" + new Date().getTime(); // force reload iframe source without browser cache
                mapFrame.classList.remove('hidden');
                loadingState.classList.add('hidden');
            } else {
                alert('Generation failed: ' + data.message);
                loadingState.classList.add('hidden');
                idleState.classList.remove('hidden');
            }

        } catch (err) {
            console.error(err);
            alert('An error occurred connecting to the backend server.');
            loadingState.classList.add('hidden');
            idleState.classList.remove('hidden');
        } finally {
            // Restore button
            btn.disabled = false;
            btnText.textContent = 'Generate Routes';
            btnSpinner.classList.add('hidden');
        }
    });

});
