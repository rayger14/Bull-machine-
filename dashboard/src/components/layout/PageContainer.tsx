import { Outlet } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import { useAppStore } from '../../stores/useAppStore';
import { pageTransition } from '../../theme/animations';

export default function PageContainer() {
  const location = useLocation();
  const sidebarExpanded = useAppStore((s) => s.sidebarExpanded);

  return (
    <main
      className={`min-h-screen transition-all duration-300 ${
        sidebarExpanded ? 'ml-60' : 'ml-16'
      }`}
    >
      <div className="p-6 max-w-[1600px] mx-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            variants={pageTransition}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  );
}
