import { create } from 'zustand';

interface AppStore {
  sidebarExpanded: boolean;
  toggleSidebar: () => void;
  expandedSignalId: string | null;
  setExpandedSignalId: (id: string | null) => void;
  expandedTradeId: string | null;
  setExpandedTradeId: (id: string | null) => void;
  expandedPositionId: string | null;
  setExpandedPositionId: (id: string | null) => void;
  showRiskDetail: boolean;
  showInstabDetail: boolean;
  showCrisisDetail: boolean;
  toggleRiskDetail: () => void;
  toggleInstabDetail: () => void;
  toggleCrisisDetail: () => void;
  expandedStress: string;
  setExpandedStress: (s: string) => void;
  showOutlookDetail: string;
  setShowOutlookDetail: (s: string) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  sidebarExpanded: true,
  toggleSidebar: () => set((s) => ({ sidebarExpanded: !s.sidebarExpanded })),
  expandedSignalId: null,
  setExpandedSignalId: (id) => set({ expandedSignalId: id }),
  expandedTradeId: null,
  setExpandedTradeId: (id) => set({ expandedTradeId: id }),
  expandedPositionId: null,
  setExpandedPositionId: (id) => set({ expandedPositionId: id }),
  showRiskDetail: false,
  showInstabDetail: false,
  showCrisisDetail: false,
  toggleRiskDetail: () => set((s) => ({ showRiskDetail: !s.showRiskDetail })),
  toggleInstabDetail: () => set((s) => ({ showInstabDetail: !s.showInstabDetail })),
  toggleCrisisDetail: () => set((s) => ({ showCrisisDetail: !s.showCrisisDetail })),
  expandedStress: '',
  setExpandedStress: (s) => set({ expandedStress: s }),
  showOutlookDetail: '',
  setShowOutlookDetail: (s) => set({ showOutlookDetail: s }),
}));
