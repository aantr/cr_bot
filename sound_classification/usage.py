def main():
    # Step 1: Pre-calculate your sound profiles (do once)
    precalculator = SoundPrecalculator(device='cuda')
    
    sound_files = {
        'alarm': 'sounds/alarm.wav',
        'doorbell': 'sounds/doorbell.wav',
        'glass_break': 'sounds/glass_break.wav',
        # Add your 3-second sounds here
    }
    
    profiles = []
    for name, path in sound_files.items():
        profile = precalculator.precalculate_sound(path, name)
        profiles.append(profile)
    
    precalculator.save_profiles(profiles, 'sound_profiles.pkl')
    print("Profiles pre-calculated and saved")
    
    # Step 2: Train model (if you have labeled training data)
    # train_model('training_data/')
    
    # Step 3: Start real-time detection
    detector = RealtimeDetector(
        model_path='best_model.pth',
        profiles_path='sound_profiles.pkl'
    )
    
    detector.start()
    
    try:
        while True:
            if not detector.output_queue.empty():
                detection = detector.output_queue.get()
                print(f"Detected: Class {detection['class_id']} "
                      f"(Confidence: {detection['confidence']:.3f})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        detector.stop()
        print("Detection stopped")

if __name__ == "__main__":
    main()