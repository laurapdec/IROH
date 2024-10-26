# micromixing/curl_model.py

class CurlModel:
    def apply_mixing(self, particle_a, particle_b):
        for prop in particle_a.properties:
            # Average between two particles (assuming particles a and b are interacting)
            avg = (particle_a.properties[prop] + particle_b.properties[prop]) / 2
            particle_a.properties[prop] = avg
            particle_b.properties[prop] = avg
